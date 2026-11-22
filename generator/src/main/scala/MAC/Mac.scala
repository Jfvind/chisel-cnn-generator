package mac

import chisel3._
import chisel3.util._

// One multiply-accumulate unit: k multipliers, an adder and one accumulator register.
// Gets k (a, w) pairs per clock and keeps the running sum in acc, so it never needs all inputs at once.
//
// Three stages, so the multiply, the product sum and the accumulate are never in the same clock:
//   clock t:   prod(j) := a(j) * w(j)          product register = the DSP48's own MREG, no LUTs
//   clock t+1: sum     := sum_j prod(j)
//   clock t+2: acc     += sum
// A product shows up in acc Mac.Latency clocks after en. A new (a, w) can still come every clock.
class Mac(width: Int, accW: Int, k: Int = 1) extends Module {
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))   // activations, unsigned (output of ReLU or pixels)
        val w     = Input(Vec(k, SInt(width.W)))   // weights, signed
        val clear = Input(Bool())                  // acc := 0 and drops the products in flight; wins over en
        val en    = Input(Bool())                  // acc := acc + sum_j a(j) * w(j), Mac.Latency clocks later
        val acc   = Output(SInt(accW.W))
    })

    // UInt * SInt zero-extends a, so 255 stays positive
    val prod = RegNext(VecInit(io.a.zip(io.w).map { case (a, w) => a * w }))
    val v1   = RegNext(io.en && !io.clear, false.B)
    val sum  = RegNext(prod.reduce(_ +& _))              // +& keeps the carry of the product sum
    val v2   = RegNext(v1 && !io.clear, false.B)

    val acc = RegInit(0.S(accW.W))
    when(io.clear) {
        acc := 0.S
    }.elsewhen(v2) {
        acc := acc + sum
    }

    io.acc := acc
}

object Mac {
    val Latency = 3     // clocks from en (with a, w) to the product being in acc
}

// Two lanes in one set of multipliers: lanes n and n+1 get the same activation, so one DSP48E1
// (25 x 18 signed multiplier) computes both products at once:
//
//   P = a * (u1 * 2^S + u0) = a*u1 * 2^S + a*u0,   S = 2 * width,  u0, u1 = w0, w1 read as unsigned (0..255)
//
// The packed operand is only wires: Cat(w1, zeros, w0), 3 * width = 24 bits for 8-bit weights.
// 0 <= a*u0 < 2^S, so the low S bits of P are exactly a*u0 and the high bits exactly a*u1: nothing carries
// between the halves. a*w = a*u - 2^width * a when w is negative; that correction is summed beside the
// multiply and subtracted once per clock. Same stages and Latency as Mac.
//
// Everything in the multiply is unsigned on purpose: Chisel emits a signed multiply as an unsigned one on
// sign-extended operands, which Vivado can no longer fit into one DSP (it then takes two DSPs or LUTs).
// 24 x 8 unsigned fits the 25 x 18 signed DSP multiplier.
class MacPair(width: Int, accW: Int, k: Int = 1) extends Module {
    require(MacPair.fits(width), s"$width-bit weights do not pack two per DSP48E1 (needs 3 x width <= 25)")
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))
        val w0    = Input(Vec(k, SInt(width.W)))   // weights of the first lane
        val w1    = Input(Vec(k, SInt(width.W)))   // weights of the second lane
        val clear = Input(Bool())
        val en    = Input(Bool())
        val acc0  = Output(SInt(accW.W))
        val acc1  = Output(SInt(accW.W))
    })

    val S = 2 * width

    // --- clock t: packed multiply (DSP, product register = MREG) + the sign correction beside it ---
    val packed = io.w0.zip(io.w1).map { case (w0, w1) => Cat(w1.asUInt, 0.U((S - width).W), w0.asUInt) }
    val prod   = RegNext(VecInit(packed.zip(io.a).map { case (p, a) => p * a }))   // UInt x UInt
    // sum of the activations whose weight is negative, per lane
    def corrOf(ws: Vec[SInt]): UInt = io.a.zip(ws).map { case (a, w) => Mux(w(width - 1), a, 0.U) }.reduce(_ +& _)
    val corr0  = RegNext(corrOf(io.w0))
    val corr1  = RegNext(corrOf(io.w1))
    val v1     = RegNext(io.en && !io.clear, false.B)

    // --- clock t+1: unpack and sum the k products of each lane ---
    val lo   = prod.map(_(S - 1, 0))                     // a * u0
    val hi   = prod.map(_ >> S)                          // a * u1
    val sum0 = RegNext(lo.reduce(_ +& _).zext -& (corr0 << width).zext)
    val sum1 = RegNext(hi.reduce(_ +& _).zext -& (corr1 << width).zext)
    val v2   = RegNext(v1 && !io.clear, false.B)

    // --- clock t+2: accumulate ---
    val acc0 = RegInit(0.S(accW.W))
    val acc1 = RegInit(0.S(accW.W))
    when(io.clear) {
        acc0 := 0.S
        acc1 := 0.S
    }.elsewhen(v2) {
        acc0 := acc0 + sum0
        acc1 := acc1 + sum1
    }

    io.acc0 := acc0
    io.acc1 := acc1
}

object MacPair {
    // DSP48E1: the packed weight (3 x width bits) must fit the 25-bit A input, a (width + 1 bits signed) the 18-bit B
    def fits(width: Int): Boolean = 3 * width <= 25 && width + 1 <= 18
}
