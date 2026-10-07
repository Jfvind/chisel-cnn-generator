package activation

import chisel3._
import chisel3.util._

class Activation(width: Int, activations: Int, shift: Int) extends Module {
  val sumMaxSize = width * 2 + log2Ceil(activations)

  val io = IO(new Bundle {
    val sum = Input(SInt(sumMaxSize.W))
    val act = Output(UInt(width.W))
  })

  val relu = Mux(io.sum < 0.S, 0.U, io.sum.asUInt)

  val shifted = relu >> shift // We need to divide the sum with some 2^shift value, to quant
                              // it down to fit inside the output register

  io.act := shifted(width - 1, 0)
}