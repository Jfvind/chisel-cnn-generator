package mlp

import chisel3._
import chisel3.util._
import model.LayerParams

// =============================================================================
// MlpIO.scala — the contract between MlpCore (shell side) and MlpDatapath (compute side).
//
// MlpParams     : everything both sides derive from the quantized layers (sizes, widths,
//                 ROM depth/width/file name). Built by ShellGen from the NetworkModel.
// InBufReadIO   : how the datapath reads the input pixels that MlpCore buffered.
// MlpDatapathIO : start/done handshake, final scores and the overflow counter.
// MlpDatapath   : abstract base; the real datapath (and a fake one for tests) extends it.
//
// Rules:
//   * Weight ROM per layer, file romFile(l). Address j * words(in) + k holds the weights
//     of neuron j for inputs k*mnc .. k*mnc+mnc-1. Lane m sits in bits [(m+1)*w-1 : m*w],
//     two's complement. Lanes past the last input hold 0. With mnc = 1: address j*in + i.
//   * Both the ROMs and the input buffer have ONE cycle read latency.
//   * Bias = constants (VecInit of l.bias), shift = Scala constant (l.shift). No ROM.
//   * Hidden layers: ReLU, shift, saturate. Last layer: identity (raw signed sum).
//   * Argmax is done in MlpCore (lowest index wins ties), not in the datapath.
// =============================================================================

case class MlpParams(
    layers: Seq[LayerParams],   // quantized layers from WeightLoader, in order
    weightBits: Int = 8,        // signed weights
    actBits: Int = 8,           // unsigned input pixels and hidden activations
    mnc: Int = 1                // MACs per Neuron per Cycle: inputs consumed per read
) {
    require(layers.nonEmpty, "MLP needs at least one layer")
    require(mnc >= 1, s"mnc must be >= 1, got $mnc")

    val nIn  = layers.head.inFeatures       // beats in  (784 for MNIST)
    val nOut = layers.last.outFeatures      // beats out (10 classes)

    // Signed MAC sum of one layer: product width + room to add inFeatures products
    def accBits(l: LayerParams): Int = weightBits + actBits + log2Ceil(l.inFeatures)
    val scoreBits = accBits(layers.last)    // raw last-layer sums (23 bit for 128 inputs @ 8/8)

    // Reads needed to cover a layer with inFeatures inputs, mnc at a time (last read zero-padded)
    def words(inFeatures: Int): Int = (inFeatures + mnc - 1) / mnc

    def romDepth(l: LayerParams): Int    = l.outFeatures * words(l.inFeatures)
    def romWidth: Int                    = mnc * weightBits
    def romFile(l: LayerParams): String  = s"l${l.index}_w.mem"

    // Address width for n entries (at least 1 bit, so a 1-entry memory still gets a UInt(1.W))
    def addrBits(n: Int): Int = log2Ceil(n.max(2))
}

// Datapath reads input pixels from MlpCore's input buffer, mnc lanes per read.
class InBufReadIO(p: MlpParams) extends Bundle {
    val en   = Output(Bool())
    val addr = Output(UInt(p.addrBits(p.words(p.nIn)).W))
    val data = Input(Vec(p.mnc, UInt(p.actBits.W)))        // valid ONE cycle after en/addr
}

class MlpDatapathIO(p: MlpParams) extends Bundle {
    val start  = Input(Bool())                               // 1-cycle pulse, after LOAD is complete
    val in     = new InBufReadIO(p)
    val done   = Output(Bool())                              // 1-cycle pulse
    val scores = Output(Vec(p.nOut, SInt(p.scoreBits.W)))    // valid from done until next start
    val ovf    = Output(UInt(32.W))                          // activations that saturated
}

abstract class MlpDatapath(val p: MlpParams) extends Module {
    val io = IO(new MlpDatapathIO(p))
}