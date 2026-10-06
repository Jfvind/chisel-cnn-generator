package mac

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class MacArrayTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "MacArray"

    val width = 8
    val prevLayerSize   = 4
    val min   = -(1 << (width - 1))    // -128
    val max   = (1 << (width - 1)) - 1 //  127

    def check(dut: MacArray, acts: Seq[Int], weights: Seq[Int], bias: Int): Unit = {
        for (i <- 0 until n) {
            dut.io.acts(i).poke(acts(i).S)
            dut.io.weights(i).poke(weights(i).S)
        }
        dut.io.bias.poke(bias.S)
        val expected = acts.zip(weights).map { case (a, w) => a * w }.sum + bias
        dut.io.sum.expect(expected.S)
    }

    it should "compute a simple dot product" in {
        test(new MacArray(width, n)) { dut =>
            check(dut, Seq(1, 2, 3, 4),   Seq(1, 1, 1, 1), 0) // 10
            check(dut, Seq(1, -2, 3, -4), Seq(2, 2, 2, 2), 5) // -4 + 5 = 1
        }
    }

    it should "not overflow at extreme values" in {
        test(new MacArray(width, n)) { dut =>
            check(dut, Seq.fill(n)(min), Seq.fill(n)(min), max) // largest positive sum
            check(dut, Seq.fill(n)(min), Seq.fill(n)(max), min) // largest negative sum
        }
    }

    it should "match a software model on random inputs" in {
        val rnd = new Random(42)
        def r() = rnd.nextInt(max - min + 1) + min
        test(new MacArray(width, n)) { dut =>
            for (_ <- 0 until 100) {
                check(dut, Seq.fill(n)(r()), Seq.fill(n)(r()), r())
            }
        }
    }
}