package mac

import chisel3._

// The MAC column: lanes x Mac. Every lane gets the same k activations a (broadcast) and its own k weights,
// so one clock feeds k inputs to all neurons of a layer at once.
class MacArray(width: Int, lanes: Int, accW: Int, k: Int = 1) extends Module {
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))
        val w     = Input(Vec(lanes, Vec(k, SInt(width.W))))  // w(n) goes to lane n
        val clear = Input(Bool())
        val en    = Input(Bool())
        val acc   = Output(Vec(lanes, SInt(accW.W)))
    })

    val macs = Seq.fill(lanes)(Module(new Mac(width, accW, k)))

    for ((m, n) <- macs.zipWithIndex) {
        m.io.a     := io.a
        m.io.w     := io.w(n)
        m.io.clear := io.clear
        m.io.en    := io.en
        io.acc(n)  := m.io.acc
    }
}
