package mlp

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFileInline
import java.io.File
import java.nio.file.Files

import activation.Activation
import mac.MacArray
import model.NetworkModel
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
// Widths come from net.quantization (Mac and Activation use one width for weights and activations).
// OVF_COUNT = hidden activations that saturated since the last CTRL start (0 with shift policy "worst").
//
// Weight bank layout (one SyncReadMem per lane, initialised from bank_<n>.hex at elaboration):
//   row layerBase(l) + i = w_l[i][n],  layerBase(l) = sum of inFeatures of the layers before l,
//   0 when lane n is unused in layer l. MNIST 784 -> 128 -> 10: 912 rows x 8 bits per lane.
//
// Pipeline: bank read (1 cycle) + bank output register (1 cycle), the input is registered twice to meet
// its weight; the MACs accumulate two cycles after the read is issued.
//
// Cycles at full speed (stream never stalls): first s_axis beat at cycle 0, last m_axis beat at
// cycle  sum_l (in_l + 4) + nOut - 1   (in_l reads, 2 cycles to the last MAC, sum + finish per layer).
// MNIST: (784 + 4) + (128 + 4) + 10 - 1 = 929.
// =============================================================================

class MlpCore(cfg: AccelConfig, net: NetworkModel) extends AccelCore(cfg) {
    val q       = net.quantization
    require(q.weightBits == q.activationBits,
        s"MlpCore needs equal weight/activation widths (Mac, Activation), got ${q.weightBits}/${q.activationBits}")
    val Bits    = q.activationBits                     // weight and activation width
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
        val digits = (Bits + 3) / 4                  // two's complement in Bits bits: -2 @ 8 bit -> "fe"
        Files.write(file.toPath, rows.map(w => String.format(s"%0${digits}x", Int.box(w & ((1 << Bits) - 1))))
            .mkString("", "\n", "\n").getBytes)
        val mem = SyncReadMem(depth, UInt(Bits.W))   // stored as raw bits; a SInt memory loads "fe" as 254 in treadle
        // forward slashes: a Windows path "C:\Users\..." inside a Verilog string would be read as escape sequences
        loadMemoryFromFileInline(mem, file.getAbsolutePath.replace('\\', '/'))
        mem
    }

    // --- Batching input images ---
    val batch = RegInit(1.U(32.W))              // Images pr batch. Default: 1
    val img   = RegInit(0.U(32.W))              // Image index in the batch
    when (io.start) {
        batch := Mux(io.args(0) === 0.U, 1.U, io.args(0))
    }
    val lastImage = img === batch - 1.U

    // --- Control state ---
    // run -> sum (acc + bias into registers) -> finish (activation or logits) -> run / output
    object State extends ChiselEnum { val run, sum, finish, output = Value }
    val state   = RegInit(State.run)
    val layer   = RegInit(0.U(log2Ceil(layers.length max 2).W))
    val i       = RegInit(0.U(log2Ceil(maxIn max N max 2).W))  // input index within the layer
    val drained = RegInit(false.B)                              // last input issued, waiting for the MAC
    val o       = RegInit(0.U(log2Ceil(cfg.nOut max 2).W))      // output beat index

    val act    = Reg(Vec(N, UInt(Bits.W)))                      // activation registers (hidden layers)
    val logits = Reg(Vec(cfg.nOut, SInt(accW.W)))               // logit registers (last layer)
    val sumReg = Reg(Vec(N, SInt(accW.W)))                      // pipeline: acc + bias is one cycle before activation
    val result = RegInit(0.U(32.W))

    val inLast    = pick(layers.map(l => (l.inFeatures - 1).U), layer)
    val base      = pick(layerBase.init.map(_.U(log2Ceil(depth max 2).W)), layer)
    val shift     = pick(layers.map(_.shift.U(log2Ceil(accW).W)), layer)
    val lastLayer = layer === (layers.length - 1).U

    // --- Input: ppb pixels per stream word (lowest bits = first pixel); later layers from the activation registers ---
    val ppb        = cfg.pixelsPerBeat                          // pixels per beat (e.g. 32 / 8 = 4)
    require(ppb >= 1 && ppb * Bits <= cfg.streamW,
        s"pixelsPerBeat $ppb x $Bits bit does not fit a ${cfg.streamW}-bit stream word")
    val word       = Reg(UInt(cfg.streamW.W))                   // the word being unpacked
    val have       = RegInit(0.U(log2Ceil(ppb + 1).W))          // pixels left in `word`
    val fromStream = layer === 0.U
    val canIssue   = state === State.run && !drained
    val pixValid   = have =/= 0.U || io.in.valid
    val pixel      = Mux(have =/= 0.U, word(Bits - 1, 0), io.in.bits.data(Bits - 1, 0))
    io.in.ready   := canIssue && fromStream && have === 0.U      // fetch a new word only when the old one is used
    val issue      = canIssue && (!fromStream || pixValid)
    val actIdx     = if (N == 1) 0.U else i(log2Ceil(N) - 1, 0)
    val a          = Mux(fromStream, pixel, act(actIdx))

    when(issue && fromStream) {
        val endOfImage = i === inLast                            // the rest of a partly used word is padding
        when(have === 0.U) {                                     // took pixel 0 straight from the stream
            word := io.in.bits.data >> Bits
            have := Mux(endOfImage, 0.U, (ppb - 1).U)
        }.otherwise {
            word := word >> Bits
            have := Mux(endOfImage, 0.U, have - 1.U)
        }
    }

    when(issue) {
        i := Mux(i === inLast, 0.U, i + 1.U)
        when(i === inLast) { drained := true.B }
    }

    // One pipeline stage for a valid/control bit; CTRL start empties it, so nothing from an
    // aborted image reaches the MACs or the state machine after a restart
    def stage(b: Bool): Bool = RegNext(b && !io.start, false.B)

    // --- MAC column, one cycle behind the issue (bank read latency) ---
    val macs = Module(new MacArray(Bits, N, accW))
    for (n <- 0 until N) macs.io.w(n) := RegNext(banks(n).read(base + i, issue).asSInt)
    macs.io.a     := RegNext(RegNext(a))                                 // pipeline register
    macs.io.en    := stage(stage(issue))
    macs.io.clear := false.B
    when(stage(stage(issue && i === inLast))) { state := State.sum }   // last product is in acc now

    // --- sum: acc + bias into registers ---
    when(state === State.sum) {
        for (n <- 0 until N) {
            val bias = pick(layers.map(l => (if (n < l.outFeatures) l.bias(n) else 0).S(accW.W)), layer)
            sumReg(n) := macs.io.acc(n) + bias
        }
        macs.io.clear := true.B
        drained       := false.B
        state         := State.finish
    }

    // --- End of layer: activation from registered sum (hidden) or raw logits (last layer) ---
    val units = sumReg.map { s =>
        val u = Module(new Activation(Bits, maxIn))
        u.io.sum   := s
        u.io.shift := shift
        u
    }

    // OVF_COUNT: saturated hidden activations since the last CTRL start (unused lanes have sum 0, never clamp)
    val ovf = RegInit(0.U(32.W))
    val ovfFlags = RegNext(VecInit(units.map(_.io.clamped)))
    val ovfValid = stage(state === State.finish && !lastLayer)
    val ovfInc   = RegNext(PopCount(ovfFlags))
    val ovfAdd   = stage(ovfValid)
    when(io.start) {
        ovf := 0.U
    }.elsewhen(ovfAdd) {
        ovf := ovf + ovfInc
    }

    when(state === State.finish) {
        when(lastLayer) {
            for (k <- 0 until cfg.nOut) logits(k) := sumReg(k)
            state  := State.output
        }.otherwise {
            for (n <- 0 until N) act(n) := units(n).io.act
            layer := layer + 1.U
            state := State.run
        }
    }

    // --- Output: nOut logits, TLAST on the last one ---
    val cur      = pick(logits, o)
    val bestVal  = Reg(SInt(accW.W))
    val bestIdx  = Reg(UInt(32.W))
    val isBetter = o === 0.U || cur > bestVal           // Only greater. lowest index wins ties
    val lastBeat = o === (cfg.nOut - 1).U               // Last score of this image
    io.out.valid     := state === State.output
    io.out.bits.data := saturate(cur)
    io.out.bits.last := lastBeat && lastImage
    when(io.out.fire) {
        when(isBetter) { bestVal := cur; bestIdx := o }
        o := Mux(lastBeat, 0.U, o + 1.U)
        when(lastBeat) { 
            result := Mux(isBetter, o, bestIdx)     // RESULT valid the cycle after done
            img    := Mux(lastImage, 0.U, img + 1.U)
            state  := State.run
            layer  := 0.U                           // ready for the next input 
        }
    }

    // --- CTRL start: every run begins from a clean state, also after an aborted batch ---
    when(io.start) {
        state         := State.run
        layer         := 0.U
        i             := 0.U
        drained       := false.B
        o             := 0.U
        img           := 0.U
        have          := 0.U                             // drops a half-used stream word; `word` is then unused
        macs.io.clear := true.B                          // partial sums of an aborted image
    }


    io.done   := io.out.fire && io.out.bits.last
    io.result := result
    io.busy   := !(state === State.run && layer === 0.U && i === 0.U && !drained)
    io.ovf    := ovf

    // xs(idx) for a per-layer (or per-beat) constant; a one-element Vec takes no index bits
    def pick[T <: Data](xs: Seq[T], idx: UInt): T = if (xs.length == 1) xs.head else VecInit(xs)(idx)

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
