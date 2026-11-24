package formal

import chisel3._
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec

import mac.{Mac, MacPair}

// The DSP packing is exact. MacPair (two lanes in one packed multiplier, a * (w1 * 2^16 + w0) with a
// sign correction) must give the same accumulators as two plain Macs, for ALL inputs, not just random ones.
class MacPairVsMacs(width: Int, accW: Int, k: Int, oneProduct: Boolean) extends Module {
    val io = IO(new Bundle {
        val a     = Input(Vec(k, UInt(width.W)))
        val w0    = Input(Vec(k, SInt(width.W)))
        val w1    = Input(Vec(k, SInt(width.W)))
        val clear = Input(Bool())
        val en    = Input(Bool())
    })
    val pair = Module(new MacPair(width, accW, k))
    val m0   = Module(new Mac(width, accW, k))
    val m1   = Module(new Mac(width, accW, k))
    pair.io.a := io.a; pair.io.w0 := io.w0; pair.io.w1 := io.w1; pair.io.clear := io.clear; pair.io.en := io.en
    m0.io.a   := io.a; m0.io.w := io.w0; m0.io.clear := io.clear; m0.io.en := io.en
    m1.io.a   := io.a; m1.io.w := io.w1; m1.io.clear := io.clear; m1.io.en := io.en

    // 8-bit multipliers are slow to prove: let exactly one product in (first cycle after reset), any value
    if (oneProduct) {
        val first = RegInit(true.B)
        first := false.B
        chisel3.assume(io.en === first && !io.clear, "one product, no clear")
    }

    chisel3.assert(pair.io.acc0 === m0.io.acc, "packed lane 0 differs from a plain Mac")
    chisel3.assert(pair.io.acc1 === m1.io.acc, "packed lane 1 differs from a plain Mac")
}

class MacPairFormalSpec extends AnyFlatSpec with ChiselScalatestTester with Formal {
    behavior of "MacPair (formal)"

    // Runs only with FORMAL=1 (make formal, CI) and z3 on PATH; otherwise canceled, so `make test` stays fast
    def skipUnlessFormal(): Unit = {
        if (!sys.env.get("FORMAL").contains("1")) cancel("runs with `make formal` (FORMAL=1)")
        val z3 = try { scala.sys.process.Process(Seq("z3", "-version")).!!; true } catch { case _: Throwable => false }
        if (!z3) cancel("z3 not found on PATH (pip install z3-solver)")
    }

    it should "equal two plain Macs for any 8-bit activation and weights" in {
        Z3.skip.foreach(cancel(_))
        verify(new MacPairVsMacs(8, 26, 1, oneProduct = true), Seq(BoundedCheck(6)))
    }

    it should "equal two plain Macs for any 4-bit inputs, en and clear sequence" in {
        Z3.skip.foreach(cancel(_))
        verify(new MacPairVsMacs(4, 12, 1, oneProduct = false), Seq(BoundedCheck(5)))
    }
}
