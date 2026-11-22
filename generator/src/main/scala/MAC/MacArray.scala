package mac

import chisel3._

// The MAC column: lanes x Mac. Every lane gets the same k activations a (broadcast) and its own k weights,
// so one clock feeds k inputs to all neurons of a layer at once.
// pack = true: two neighbouring lanes share one multiplier per input (MacPair), which halves the DSPs.
// An odd last lane is paired with a lane of weight 0. Both ways have latency Mac.Latency.
class MacArray(width: Int, lanes: Int, accW: Int, k: Int = 1, pack: Boolean = false) extends Module {
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))
        val w     = Input(Vec(lanes, Vec(k, SInt(width.W))))  // w(n) goes to lane n
        val clear = Input(Bool())
        val en    = Input(Bool())
        val acc   = Output(Vec(lanes, SInt(accW.W)))
    })

    if (pack) {
        val zero = VecInit(Seq.fill(k)(0.S(width.W)))
        for (p <- 0 until (lanes + 1) / 2) {
            val m = Module(new MacPair(width, accW, k))
            m.io.a     := io.a
            m.io.w0    := io.w(2 * p)
            m.io.w1    := (if (2 * p + 1 < lanes) io.w(2 * p + 1) else zero)
            m.io.clear := io.clear
            m.io.en    := io.en
            io.acc(2 * p) := m.io.acc0
            if (2 * p + 1 < lanes) io.acc(2 * p + 1) := m.io.acc1
        }
    } else {
        for (n <- 0 until lanes) {
            val m = Module(new Mac(width, accW, k))
            m.io.a     := io.a
            m.io.w     := io.w(n)
            m.io.clear := io.clear
            m.io.en    := io.en
            io.acc(n)  := m.io.acc
        }
    }
}
