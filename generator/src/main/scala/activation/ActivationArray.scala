package activation

import chisel3._
import chisel3.util._

// The activation column: lanes x Activation (ReLU, >> shift, clamp), one per MAC lane, all with the same shift.
// Combinational, like Activation.
class ActivationArray(width: Int, lanes: Int, activations: Int) extends Module {
    val sumW = Activation.sumWidth(width, activations)
    val io = IO(new Bundle {
        val sum     = Input(Vec(lanes, SInt(sumW.W)))
        val shift   = Input(UInt(log2Ceil(sumW).W))
        val act     = Output(Vec(lanes, UInt(width.W)))
        val clamped = Output(Vec(lanes, Bool()))            // lane n saturated (OVF_COUNT)
    })

    for (n <- 0 until lanes) {
        val u = Module(new Activation(width, activations))
        u.io.sum        := io.sum(n)
        u.io.shift      := io.shift
        io.act(n)       := u.io.act
        io.clamped(n)   := u.io.clamped
    }
}
