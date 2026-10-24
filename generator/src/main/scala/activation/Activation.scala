package activation

import chisel3._
import chisel3.util._

class Activation(width: Int, activations: Int) extends Module {
  val sumMaxSize = width * 2 + log2Ceil(activations)
  val actMax     = ((BigInt(1) << width) - 1).U(width.W)

  val io = IO(new Bundle {
    val sum   = Input(SInt(sumMaxSize.W))
    val shift = Input(UInt(log2Ceil(sumMaxSize).W)) // read from a register, so each layer can use its own shift
    val act   = Output(UInt(width.W))
    val clamped = Output(Bool())                    // the shifted sum did not fit and was saturated (OVF_COUNT)
  })

  val relu = Mux(io.sum < 0.S, 0.U, io.sum.asUInt)

  val shifted = relu >> io.shift // We need to divide the sum with some 2^shift value, to quant
                                 // it down to fit inside the output register

  io.clamped := shifted > actMax
  io.act := Mux(io.clamped, actMax, shifted) // clamp to max instead of wrapping on overflow
}
