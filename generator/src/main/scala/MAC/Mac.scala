package mac

import chisel3._

// One multiply-accumulate unit: k multipliers, an adder and one accumulator register.
// Gets k (a, w) pairs per clock and keeps the running sum in acc, so it never needs all inputs at once.
class Mac(width: Int, accW: Int, k: Int = 1) extends Module {
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))   // activations, unsigned (output of ReLU or pixels)
        val w     = Input(Vec(k, SInt(width.W)))   // weights, signed
        val clear = Input(Bool())                  // acc := 0, wins over en
        val en    = Input(Bool())                  // acc := acc + sum_j a(j) * w(j)
        val acc   = Output(SInt(accW.W))
    })

    val acc = RegInit(0.S(accW.W))

    when(io.clear) {
        acc := 0.S
    }.elsewhen(io.en) {
        // UInt * SInt zero-extends a, so 255 stays positive; +& keeps the carry of the product sum
        acc := acc + io.a.zip(io.w).map { case (a, w) => a * w }.reduce(_ +& _)
    }

    io.acc := acc
}
