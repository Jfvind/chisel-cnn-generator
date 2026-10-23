package mlp

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFileInline
import java.io.File
import java.nio.file.Files

import activation.Activation
import mac.MacArray
import model.NetworkModel
import model.NetworkModel.Bits
import shell.{AccelConfig, AccelCore}

// =============================================================================
// MlpCore.scala — fully connected network on one shared column of MACs.
//
// N = widest layer lanes. Lane n = weight bank n + Mac n + bias n + Activation n and computes
// neuron n of whichever layer is running; the same lanes are reused for every layer.
//
// Per layer, one input per clock: input i is broadcast to all lanes, lane n reads w[i][n] from its
// bank and accumulates a[i] * w[i][n]. Layer 0 takes its inputs straight from the stream (never
// stored), later layers read the activation registers written by the layer before.
// End of a layer: hidden layer -> acc + bias -> ReLU, >> shift, clamp -> activation registers;
// last layer -> raw acc + bias (no ReLU, like train_model.py) -> logit registers -> argmax -> RESULT,
// and nOut beats out on io.out (logits saturated to streamW, TLAST on the last one).
//
// Weight bank layout (one SyncReadMem per lane, initialised from bank_<n>.hex at elaboration):
//   row layerBase(l) + i = w_l[i][n],  layerBase(l) = sum of inFeatures of the layers before l,
//   0 when lane n is unused in layer l. MNIST 784 -> 128 -> 10: 912 rows x 8 bits per lane.
//
// Pipeline: bank reads take 1 cycle, so the input is registered once (pipeline register) to meet
// its weight; the MACs accumulate the cycle after the read is issued.
//
// Cycles at full speed (stream never stalls): first s_axis beat at cycle 0, last m_axis beat at
// cycle  sum_l (in_l + 2) + nOut - 1   (in_l reads, 1 cycle for the last MAC, 1 finish cycle per layer).
// MNIST: (784 + 2) + (128 + 2) + 10 - 1 = 925.
// =============================================================================

class MlpCore(cfg: AccelConfig, net: NetworkModel) extends AccelCore(cfg) {
    val layers  = net.layers
    val N       = layers.map(_.outFeatures).max
    val maxIn   = layers.map(_.inFeatures).max
    val accW    = Bits * 2 + log2Ceil(maxIn)         // same width as Activation(Bits, maxIn).sumMaxSize
    val actMax  = (1L << Bits) - 1

    // --- Checks at elaboration ---
    require(layers.head.inFeatures == cfg.nIn, s"cfg.nIn ${cfg.nIn} != network input ${layers.head.inFeatures}")
    require(layers.last.outFeatures == cfg.nOut, s"cfg.nOut ${cfg.nOut} != network output ${layers.last.outFeatures}")
    for (l <- layers) {
        require(l.shift < accW, s"Layer ${l.index}: shift ${l.shift} does not fit the ${accW}-bit sum")
        for (j <- 0 until l.outFeatures) {      // an accumulator overflow would wrap silently, unlike the clamp
            val col = l.weights.map(_(j).toLong)
            val hi  = actMax * col.filter(_ > 0).sum + l.bias(j)
            val lo  = actMax * col.filter(_ < 0).sum + l.bias(j)
            require(hi < (1L << (accW - 1)) && lo >= -(1L << (accW - 1)),
                s"Layer ${l.index} neuron $j: worst-case sum [$lo, $hi] does not fit $accW bits")
        }
    }

    // --- Weight banks, hardcoded from the network ---
    val layerBase = layers.scanLeft(0)(_ + _.inFeatures)   // last element = total rows
    val depth     = layerBase.last
    val hexDir    = new File(f"generated/mlp-${net.hashCode}%08x").getAbsoluteFile   // absolute: Vivado's $readmemh must find it
    hexDir.mkdirs()
    val banks = Seq.tabulate(N) { n =>
        val rows = layers.flatMap(l => l.weights.map(row => if (n < l.outFeatures) row(n) else 0))
        val file = new File(hexDir, s"bank_$n.hex")
        Files.write(file.toPath, rows.map(w => f"${w & 0xff}%02x").mkString("", "\n", "\n").getBytes)
        val mem = SyncReadMem(depth, UInt(Bits.W))   // stored as raw bits; a SInt memory loads "fe" as 254 in treadle
        loadMemoryFromFileInline(mem, file.getPath)
        mem
    }

    // --- Control state ---
    object State extends ChiselEnum { val run, finish, output = Value }
    val state   = RegInit(State.run)
    val layer   = RegInit(0.U(log2Ceil(layers.length max 2).W))
    val i       = RegInit(0.U(log2Ceil(maxIn max N max 2).W))  // input index within the layer
    val drained = RegInit(false.B)                              // last input issued, waiting for the MAC
    val o       = RegInit(0.U(log2Ceil(cfg.nOut max 2).W))      // output beat index

    val act    = Reg(Vec(N, UInt(Bits.W)))                      // activation registers (hidden layers)
    val logits = Reg(Vec(cfg.nOut, SInt(accW.W)))               // logit registers (last layer)
    val result = RegInit(0.U(32.W))

    val inLast    = pick(layers.map(l => (l.inFeatures - 1).U), layer)
    val base      = pick(layerBase.init.map(_.U(log2Ceil(depth max 2).W)), layer)
    val shift     = pick(layers.map(_.shift.U(log2Ceil(accW).W)), layer)
    val lastLayer = layer === (layers.length - 1).U

    // --- Input mux + issue: layer 0 from the stream, later layers from the activation registers ---
    val fromStream = layer === 0.U
    val canIssue   = state === State.run && !drained
    io.in.ready   := canIssue && fromStream
    val issue      = canIssue && (!fromStream || io.in.valid)
    val actIdx     = if (N == 1) 0.U else i(log2Ceil(N) - 1, 0)
    val a          = Mux(fromStream, io.in.bits.data(Bits - 1, 0), act(actIdx))
    // shortcut: s_axis TLAST is not checked, the beat count comes from the network; check it if packets can be malformed

    when(issue) {
        i := Mux(i === inLast, 0.U, i + 1.U)
        when(i === inLast) { drained := true.B }
    }

    // --- MAC column, one cycle behind the issue (bank read latency) ---
    val macs = Module(new MacArray(Bits, N, accW))
    for (n <- 0 until N) macs.io.w(n) := banks(n).read(base + i, issue).asSInt
    macs.io.a     := RegNext(a)                                 // pipeline register
    macs.io.en    := RegNext(issue, false.B)
    macs.io.clear := false.B
    when(RegNext(issue && i === inLast, false.B)) { state := State.finish }   // last product is in acc now

    // --- End of layer: bias, then activation (hidden) or raw logits (last layer) ---
    val sums = Seq.tabulate(N) { n =>
        val bias = pick(layers.map(l => (if (n < l.outFeatures) l.bias(n) else 0).S(accW.W)), layer)
        macs.io.acc(n) + bias
    }
    val acts = sums.map { s =>
        val u = Module(new Activation(Bits, maxIn))
        u.io.sum   := s
        u.io.shift := shift
        u.io.act
    }

    when(state === State.finish) {
        macs.io.clear := true.B
        drained       := false.B
        when(lastLayer) {
            for (k <- 0 until cfg.nOut) logits(k) := sums(k)
            result := argmax(sums.take(cfg.nOut))
            state  := State.output
        }.otherwise {
            for (n <- 0 until N) act(n) := acts(n)
            layer := layer + 1.U
            state := State.run
        }
    }

    // --- Output: nOut logits, TLAST on the last one ---
    io.out.valid     := state === State.output
    io.out.bits.data := saturate(pick(logits, o))
    io.out.bits.last := o === (cfg.nOut - 1).U
    when(io.out.fire) {
        o := Mux(io.out.bits.last, 0.U, o + 1.U)
        when(io.out.bits.last) { state := State.run; layer := 0.U }  // ready for the next input
    }

    io.done   := io.out.fire && io.out.bits.last
    io.result := result
    io.busy   := !(state === State.run && layer === 0.U && i === 0.U && !drained)
    io.ovf    := 0.U

    // xs(idx) for a per-layer (or per-beat) constant; a one-element Vec takes no index bits
    def pick[T <: Data](xs: Seq[T], idx: UInt): T = if (xs.length == 1) xs.head else VecInit(xs)(idx)

    // Index of the largest value, first one wins on a tie (like torch.argmax)
    def argmax(xs: Seq[SInt]): UInt = xs.zipWithIndex.map { case (x, k) => (x, k.U(32.W)) }.reduceLeft { (best, next) =>
        val take = next._1 > best._1
        (Mux(take, next._1, best._1), Mux(take, next._2, best._2))
    }._2

    // Two's complement in streamW bits, clamped when the logit does not fit
    def saturate(s: SInt): UInt = {
        val w = cfg.streamW
        if (w >= accW) s.pad(w).asUInt
        else {
            val hi = ((BigInt(1) << (w - 1)) - 1).S(accW.W)
            val lo = (-(BigInt(1) << (w - 1))).S(accW.W)
            val clamped = Mux(s > hi, hi, Mux(s < lo, lo, s)).asUInt
            clamped(w - 1, 0)
        }
    }
}
