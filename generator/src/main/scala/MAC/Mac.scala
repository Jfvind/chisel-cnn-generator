package mac

import chisel3._

// One multiply-accumulate unit: k multipliers, an adder and one accumulator register.
// Gets k (a, w) pairs per clock and keeps the running sum in acc, so it never needs all inputs at once.
//
// Two stages, so the multiply and the add are never in the same clock (timing at 100 MHz):
//   clock t:   prod(j) := a(j) * w(j)          product register = the DSP48's own MREG, no LUTs
//   clock t+1: acc     += sum_j prod(j)
// A product therefore shows up in acc two clocks after en, not one. A new (a, w) can still come every clock.
class Mac(width: Int, accW: Int, k: Int = 1) extends Module {
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))   // activations, unsigned (output of ReLU or pixels)
        val w     = Input(Vec(k, SInt(width.W)))   // weights, signed
        val clear = Input(Bool())                  // acc := 0 and drops the product in flight; wins over en
        val en    = Input(Bool())                  // acc := acc + sum_j a(j) * w(j), one clock later
        val acc   = Output(SInt(accW.W))
    })

    // UInt * SInt zero-extends a, so 255 stays positive
    val prod  = RegNext(VecInit(io.a.zip(io.w).map { case (a, w) => a * w }))
    val valid = RegNext(io.en && !io.clear, false.B)     // prod holds a product that still has to be added

    val acc = RegInit(0.S(accW.W))

    when(io.clear) {
        acc := 0.S
    }.elsewhen(valid) {
        acc := acc + prod.reduce(_ +& _)                 // +& keeps the carry of the product sum
    }

    io.acc := acc
}
