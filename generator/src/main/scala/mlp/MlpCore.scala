package mlp

import chisel3._
import chisel3.util._

import activation.{Activation, ActivationArray}
import mac.{Mac, MacArray}
import model.NetworkModel
import shell.{AccelConfig, AccelCore}

// =============================================================================
// MlpCore.scala — fully connected network on one shared column of MACs.
//
// Structure (FSM with datapath)
//   MlpCore         : the controller. State machine run -> sum -> finish -> run (next layer) / output -> run
//                     (next image), the layer / input / image counters, and the wiring below.
//   StreamReader    : unpacks K pixels per clock from the input stream, checks the packet length (TLAST).
//   WeightBanks     : one weight memory per lane, written from the network at elaboration (bank_<n>.hex).
//   MacArray        : N lanes multiply-accumulate K inputs per clock.
//   ActivationArray : N x ReLU, >> shift, clamp at the end of a hidden layer.
//   OutputArgmax    : sends the logits as stream beats and tracks the argmax -> RESULT.
//
// Lanes
//   N = widest layer. Lane n = weight bank n + Mac n + bias n + Activation n and computes neuron n
//   of whichever layer is running; the same lanes are reused for every layer.
//
// Dataflow (per layer, K inputs per clock)
//   Inputs Ki..Ki+K-1 are broadcast to all lanes; lane n reads their K weights from one bank row and
//   accumulates sum_j a[Ki+j] * w[Ki+j][n]. Layer 0 takes its inputs straight from the stream (never
//   stored); later layers read the activation registers written by the layer before.
//   End of a hidden layer: acc + bias -> ReLU, >> shift, clamp -> activation registers.
//   End of the last layer: raw acc + bias (no ReLU, like train_model.py) -> logit registers -> OutputArgmax.
//   Widths come from net.quantization (Mac and Activation use one width for weights and activations).
//   OVF_COUNT = hidden activations that saturated since the last CTRL start (0 with shift policy "worst").
//
// Multipliers (K and packed: Parallelism, from cfg.inputsPerClock, cfg.multiplier and cfg.dspBudget)
//   packed: lanes 2m and 2m+1 share one DSP48E1 per input (MacPair: a * (w1 * 2^16 + w0) gives both
//   products), so N lanes x K inputs need N / 2 * K DSPs. single: one multiplier per lane and input (Mac).
//   MNIST, packed, K = 4: 64 x 4 = 256 multipliers.
//
// Pipeline
//   WeightBanks latency 2 (read + output register); the input is registered twice to meet its weight.
//   The MACs multiply two cycles after the read is issued, and the product is in the accumulator
//   Mac.Latency = 3 cycles after that (multiply, sum of the K products, accumulate: one per clock).
//
// Timing at full speed (stream never stalls)
//   First s_axis beat at cycle 0, last m_axis beat at cycle  sum_l (ceil(in_l / K) + 6) + nOut - 1
//   (ceil(in_l / K) reads, 2 cycles to the last multiply, 2 to the last accumulate, sum + finish per layer).
//   MNIST, K = 4: (196 + 6) + (32 + 6) + 10 - 1 = 249.
// =============================================================================

class MlpCore(cfg: AccelConfig, net: NetworkModel) extends AccelCore(cfg) {
    import MlpCore.pick

    val q       = net.quantization
    require(q.weightBits == q.activationBits,
        s"MlpCore needs equal weight/activation widths (Mac, Activation), got ${q.weightBits}/${q.activationBits}")
    val Bits    = q.activationBits                      // weight and activation width
    val layers  = net.layers
    val N       = layers.map(_.outFeatures).max         // Number of neurons in the biggest layer
    val maxIn   = layers.map(_.inFeatures).max          // Most weights per neuron in any layer (= most inputs)
    val accW    = Activation.sumWidth(Bits, maxIn)      // accumulator width
    val ppb     = cfg.pixelsPerBeat                     // pixels per stream word (e.g. 32 / 8 = 4)
    require(ppb * Bits <= cfg.streamW,
        s"pixelsPerBeat $ppb x $Bits bit does not fit a ${cfg.streamW}-bit stream word")
    val parallelism = Parallelism.choose(cfg, net)
    val pack    = parallelism.packed                    // two lanes per DSP (MacPair)
    val K       = parallelism.inputsPerClock            // inputs per clock; divides ppb so a group never spans two words
    val groups  = layers.map(l => (l.inFeatures + K - 1) / K)   // bank rows (and clocks) per layer
    val layerBase = groups.scanLeft(0)(_ + _)           // first bank row of each layer; last element = total rows

    // --- Checks at elaboration ---
    require(layers.head.inFeatures == cfg.nIn, s"cfg.nIn ${cfg.nIn} != network input ${layers.head.inFeatures}")
    require(layers.last.outFeatures == cfg.nOut, s"cfg.nOut ${cfg.nOut} != network output ${layers.last.outFeatures}")
    // Accumulator overflow is ruled out by QuantizedReader (per-layer width <= accW)
    for (l <- layers)
        require(l.shift < accW, s"Layer ${l.index}: shift ${l.shift} does not fit the ${accW}-bit sum")

    // --- Control registers ---
    object State extends ChiselEnum { val run, sum, finish, output = Value }
    import State._
    val stateReg   = RegInit(run)
    val layerReg   = RegInit(0.U(log2Ceil(layers.length max 2).W))
    val iReg       = RegInit(0.U(log2Ceil(maxIn max N max 2).W))  // input group index within the layer
    val drainedReg = RegInit(false.B)                              // last input issued, waiting for the MAC
    val batchReg   = RegInit(1.U(32.W))                            // images per batch (ARG0, 0 -> 1)
    val imgReg     = RegInit(0.U(32.W))                            // image index in the batch
    when(io.start) {
        batchReg := Mux(io.args(0) === 0.U, 1.U, io.args(0))
    }
    val lastImage = imgReg === batchReg - 1.U

    // --- Datapath registers ---
    val actReg    = Reg(Vec(N, UInt(Bits.W)))                      // activation registers (hidden layers)
    val logitsReg = Reg(Vec(cfg.nOut, SInt(accW.W)))               // logit registers (last layer)
    val sumReg    = Reg(Vec(N, SInt(accW.W)))                      // pipeline: acc + bias is one cycle before activation

    // Per-layer constants, selected by the current layer
    val inLast    = pick(groups.map(g => (g - 1).U), layerReg)
    val base      = pick(layerBase.init.map(_.U(log2Ceil(layerBase.last max 2).W)), layerReg)
    val shift     = pick(layers.map(_.shift.U(log2Ceil(accW).W)), layerReg)
    val bias      = VecInit.tabulate(N)(n => pick(layers.map(l => (if (n < l.outFeatures) l.bias(n) else 0).S(accW.W)), layerReg))
    val lastLayer = layerReg === (layers.length - 1).U

    // --- Input: layer 0 from the stream (StreamReader), later layers from the activation registers ---
    val fromStream = layerReg === 0.U
    val canIssue   = stateReg === run && !drainedReg
    // which group index fetches the last word of an image (784 pixels, 4 per word, K = 4 -> word 195 = group 195)
    val lastWordAt = (((layers.head.inFeatures - 1) / ppb) * (ppb / K)).U
    val reader = Module(new StreamReader(cfg.streamW, Bits, K, ppb))
    reader.io.in         <> io.in
    reader.io.start      := io.start
    reader.io.req        := canIssue && fromStream
    reader.io.endOfImage := iReg === inLast
    reader.io.expectEnd  := lastImage && iReg === lastWordAt
    val endBatch   = lastImage || reader.io.short                 // the image being processed is the batch's last
    val issue      = canIssue && (!fromStream || reader.io.valid)
    // act in groups of K (128 -> 32 groups of 4); the tail is padded with 0 (its weights are 0 too)
    val actPadded  = actReg.padTo((N + K - 1) / K * K, 0.U(Bits.W))
    val actGroups  = VecInit(actPadded.grouped(K).map(g => VecInit(g.toSeq)).toSeq)
    val actIdx     = if (actGroups.length == 1) 0.U else iReg(log2Ceil(actGroups.length) - 1, 0)
    val a          = Mux(fromStream, reader.io.pixels, actGroups(actIdx))

    when(issue) {
        iReg := Mux(iReg === inLast, 0.U, iReg + 1.U)
        when(iReg === inLast) { drainedReg := true.B }
    }

    // One pipeline stage for a valid/control bit; CTRL start empties it, so nothing from an
    // aborted image reaches the MACs or the state machine after a restart
    def stage(b: Bool): Bool = RegNext(b && !io.start, false.B)

    // --- Weights and MAC column: the MACs get row base + i two cycles after the issue (WeightBanks latency) ---
    val banks = Module(new WeightBanks(net, N, K, Bits, groups))
    banks.io.addr := base + iReg
    banks.io.en   := issue
    val macs = Module(new MacArray(Bits, N, accW, K, pack))
    macs.io.w     := banks.io.w
    macs.io.a     := RegNext(RegNext(a))                           // input registered twice to meet its weight
    macs.io.en    := stage(stage(issue))
    // last product is in acc: 1 + Mac.Latency stages (not ShiftRegister: CTRL start must flush it)
    val lastInAccReg = (0 until 1 + Mac.Latency).foldLeft(issue && iReg === inLast)((b, _) => stage(b))

    // --- Activation column: ReLU, >> shift, clamp of the registered sums ---
    val acts = Module(new ActivationArray(Bits, N, maxIn))
    acts.io.sum   := sumReg
    acts.io.shift := shift

    // OVF_COUNT: saturated hidden activations since the last CTRL start (unused lanes have sum 0, never clamp)
    // two register stages: clamp flags -> PopCount -> add to ovf
    val ovfReg      = RegInit(0.U(32.W))
    val ovfFlagsReg = RegNext(acts.io.clamped)
    val ovfValidReg = stage(stateReg === finish && !lastLayer)
    val ovfIncReg   = RegNext(PopCount(ovfFlagsReg))
    val ovfAddReg   = stage(ovfValidReg)
    when(io.start) {
        ovfReg := 0.U
    }.elsewhen(ovfAddReg) {
        ovfReg := ovfReg + ovfIncReg
    }

    // --- Output: nOut logits, TLAST on the last one, argmax -> RESULT ---
    val outs = Module(new OutputArgmax(cfg.nOut, accW, cfg.streamW))
    outs.io.logits   := logitsReg
    outs.io.valid    := stateReg === output
    outs.io.endBatch := endBatch
    outs.io.start    := io.start
    io.out           <> outs.io.out

    // --- State machine ---
    macs.io.clear := false.B
    switch(stateReg) {
        is(run) {                               // issue inputs; after the last one, wait for the pipeline to empty
            when(lastInAccReg) { stateReg := sum }
        }
        is(sum) {                               // acc + bias into registers, clear the MACs for the next layer
            for (n <- 0 until N) sumReg(n) := macs.io.acc(n) + bias(n)
            macs.io.clear := true.B
            drainedReg    := false.B
            stateReg      := finish
        }
        is(finish) {
            when(lastLayer) {                   // raw logits, no ReLU
                for (k <- 0 until cfg.nOut) logitsReg(k) := sumReg(k)
                stateReg := output
            }.otherwise {                       // ReLU, >> shift, clamp into the activation registers
                actReg   := acts.io.act
                layerReg := layerReg + 1.U
                stateReg := run
            }
        }
        is(output) {
            when(outs.io.lastFire) {            // last score sent: next image of the batch, or idle
                imgReg   := Mux(endBatch, 0.U, imgReg + 1.U)
                layerReg := 0.U
                stateReg := run
            }
        }
    }

    // --- CTRL start: every run begins from a clean state, also after an aborted batch (last: overrides the FSM) ---
    when(io.start) {
        stateReg      := run
        layerReg      := 0.U
        iReg          := 0.U
        drainedReg    := false.B
        imgReg        := 0.U
        macs.io.clear := true.B                 // partial sums of an aborted image
    }

    io.done   := io.out.fire && io.out.bits.last
    io.result := outs.io.result
    io.busy   := !(stateReg === run && layerReg === 0.U && iReg === 0.U && !drainedReg)
    io.ovf    := ovfReg
    io.err    := reader.io.err
}

// Helpers shared by MlpCore and its datapath modules
object MlpCore {
    // The k bits-wide fields of a packed word, field 0 in the lowest bits
    def fields(x: UInt, k: Int, bits: Int): Vec[UInt] = VecInit.tabulate(k)(j => x(j * bits + bits - 1, j * bits))

    // xs(idx) for a per-layer (or per-beat) constant; a one-element Vec takes no index bits
    def pick[T <: Data](xs: Seq[T], idx: UInt): T = if (xs.length == 1) xs.head else VecInit(xs)(idx)
}
