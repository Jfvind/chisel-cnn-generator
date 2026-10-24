package mac

import chisel3._

// One multiply-accumulate unit: one multiplier, one adder and one accumulator register.
// Gets one (a, w) pair per clock and keeps the running sum in acc, so it never needs all inputs at once.
class Mac(width: Int, accW: Int) extends Module {
    val io = IO(new Bundle {
        val a     = Input(UInt(width.W))   // activation, unsigned (output of ReLU or a pixel)
        val w     = Input(SInt(width.W))   // weight, signed
        val clear = Input(Bool())          // acc := 0, wins over en
        val en    = Input(Bool())          // acc := acc + a * w
        val acc   = Output(SInt(accW.W))
    })

    val acc = RegInit(0.S(accW.W))

    when(io.clear) {
        acc := 0.S
    }.elsewhen(io.en) {
        acc := acc + io.a * io.w           // UInt * SInt zero-extends a, so 255 stays positive
    }

    io.acc := acc
}
