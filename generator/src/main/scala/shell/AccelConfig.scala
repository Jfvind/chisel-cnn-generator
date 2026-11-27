package shell

import java.nio.file.Paths

import mlp.Parallelism
import model.NetworkModel
import service.QuantizedReader

// =============================================================================
// AccelConfig.scala — the single source of truth shared by Chisel, Vivado and PYNQ.
//
// AccelConfig : case class with core name, nIn, nOut, streamW, addrW, fclkMhz, the MLP parallelism
//               (inputsPerClock, multiplier, dspBudget; see mlp/Parallelism.scala), plus the ID magic
//               number and a hash of the config so the driver can detect a bitstream/manifest mismatch.
// RegMap      : fixed AXI-Lite register map (byte offsets), identical for every core.
//               See hw/README.md for the table.
// =============================================================================

case class AccelConfig(
    core: String = "stub",          // which core goes inside the shell
    nIn: Int = 784,                 // input beats per inference (MNIST: 28*28)
    nOut: Int = 10,                 // output beats per inference (MNIST: 0-9)
    streamW: Int = 32,              // AXI-stream width (bits)
    addrW: Int = 12,                // AXI-Lite address bit-width (2^12 = 4 KB)
    fclkMhz: Int = 100,             // PL clock requested from Zynq PS
    quantizedDir: String = "generated/quantized",   // mlp: export from `make reference`, relative to generator/
    paramHash: Long = 0,                             // set by resolve: the weights are part of CFG_HASH
    pixelsPerBeat: Int = 1,                         // set by resolve: input pixels packed per stream word (mlp: streamW / activation bits)
    inputsPerClock: Int = 0,                        // mlp: inputs per lane per clock, 0 = chosen by resolve
    multiplier: String = "auto",                    // mlp: "packed" (two lanes per DSP), "single" or "auto" = chosen by resolve
    dspBudget: Int = 0                              // mlp: most multipliers the core may build, 0 = no limit
) {
    // checks at elaboration
    require(Seq(8, 16, 32, 64).contains(streamW), s"streamW must be 8/16/32/64, got $streamW")
    require(addrW >= 8 && addrW <= 16, s"addrW must be 8-16, got $addrW")
    require(inputsPerClock >= 0, s"inputsPerClock must be positive or 0 (auto), got $inputsPerClock")
    require(Parallelism.Multipliers.contains(multiplier), s"multiplier must be auto, packed or single, got $multiplier")
    require(dspBudget >= 0, s"dspBudget must be positive or 0 (no limit), got $dspBudget")

    // Magic ID to check if correct bitstream is loaded
    val id: Long = 0xDEADBEEFL

    // Hash for checking whether AccelConfig has the expected parameters
    val cfgHash: Long = (this.toString.hashCode.toLong & 0xFFFFFFFFL) // & 0xFFFFFFFF ensures unsigned 32-bit
}

object AccelConfig {
    // mlp: read from the export, so the hardware, the test vectors and the board test share one quantization
    def network(cfg: AccelConfig): NetworkModel = QuantizedReader.load(Paths.get(cfg.quantizedDir))

    // Core-specific rules, applied by ShellGen before anything is generated, so that the
    // manifest the driver reads describes what the hardware really does.
    // stub: returns one output beat per input beat, so nOut must equal nIn.
    // mlp:  nIn/nOut come from the network, and the weights go into CFG_HASH, so a bitstream
    //       built from other weights is caught by the driver's hash check. inputsPerClock and multiplier
    //       are fixed to what Parallelism chooses and dspBudget is cleared once it has been applied, so the
    //       manifest and CFG_HASH name the built hardware: the same hardware always gets the same hash.
    def resolve(cfg: AccelConfig): AccelConfig = cfg.core match {
        case "stub" => cfg.copy(nOut = cfg.nIn)
        case "mlp"  =>
            val net = network(cfg)
            val sized = cfg.copy(nIn = net.layers.head.inFeatures, nOut = net.layers.last.outFeatures,
                     paramHash = net.hashCode.toLong & 0xFFFFFFFFL, 
                     pixelsPerBeat = (cfg.streamW / net.quantization.activationBits) max 1) // must match MlpCore ppb
            val choice = Parallelism.choose(sized, net)
            sized.copy(inputsPerClock = choice.inputsPerClock, multiplier = choice.multiplier, dspBudget = 0)
        case _      => cfg
    }
}

// Byte offset for AXI-Lite 32-bit registers
object RegMap {
    val CTRL      = 0x00 // W: bit0 = start (pulse), bit1 = soft reset of core
    val STATUS    = 0x04 // R: bit0 = busy, bit1 = done (high until reset)
    val ID        = 0x08 // R
    val CFG_HASH  = 0x0C // R
    val N_IN      = 0x10 // R
    val N_OUT     = 0x14 // R
    val CYCLES    = 0x18 // R: cycles from first s_axis beat to last m_axis beat (TLAST)
    val OVF_COUNT = 0x1C // R: saturated activations in core (test on clipping)
    val RESULT    = 0x20 // R: defined by core, (MLP: argmax)
    val ARG0      = 0x40 // RW: ARG0-ARG3, 0x40, 0x44, 0x48, 0x4C
    val NUM_ARGS  = 4
    val SUM       = 0x50 // R: ARG0 + ARG1 (adder smoke test)

    // Name and offset list written to hw_manifest.json for python use
    def offsets: Seq[(String, Int)] = Seq(
        "CTRL" ->  CTRL, "STATUS" -> STATUS, "ID" -> ID, "CFG_HASH" -> CFG_HASH,
        "N_IN" -> N_IN, "N_OUT" -> N_OUT, "CYCLES" -> CYCLES, "OVF_COUNT" -> OVF_COUNT,
        "RESULT" -> RESULT, "ARG0" -> ARG0, "SUM" -> SUM
    )
}