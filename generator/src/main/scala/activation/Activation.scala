package activation

import chisel3._
import chisel3.util._

class Activation(width: Int, activations: Int, shift: Int) extends Module {
  val sumMaxSize = width * 2 + log2Ceil(activations)

  val io = IO(new Bundle {
    val sum = Input(SInt(sumMaxSize.W))
    val act = Output(SInt(width.W))
  })

  val relu = Mux(io.sum < 0.S, 0.S, io.sum)

  val shifted = relu >> shift // We need to divide the sum with some 2^shift value, to quant
                              // it down to fit in

  // We clamp the values if the shift chosen is too small
  val maxVal  = ((BigInt(1) << (width - 1)) - 1).S
  val clamped = Mux(shifted > maxVal, maxVal, shifted)

  io.act := clamped(width - 1, 0).asSInt
}