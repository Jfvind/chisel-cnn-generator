package activation

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class ActivationTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Activation"

  val width       = 8
  val activations = 4
  val sumMaxSize  = width * 2 + chisel3.util.log2Ceil(activations) // 18 bits
  val sumMin      = -(1 << (sumMaxSize - 1))   // -131072
  val sumMax      = (1 << (sumMaxSize - 1)) - 1 //  131071
  val actMax      = (1 << (width - 1)) - 1      //  127

  // Software model: ReLU -> divide by 2^shift (rounding down) -> clamp to actMax
  def model(sum: Int, shift: Int): Int = math.min(math.max(sum, 0) >> shift, actMax)

  def check(dut: Activation, shift: Int, sum: Int): Unit = {
    dut.io.sum.poke(sum.S)
    dut.io.act.expect(model(sum, shift).S)
  }

  it should "output 0 for negative sums and zero" in {
    test(new Activation(width, activations, 4)) { dut =>
      check(dut, 4, 0)      // 0
      check(dut, 4, -1)     // 0
      check(dut, 4, -100)   // 0
      check(dut, 4, sumMin) // most negative input -> 0
    }
  }

  it should "pass small positive sums through unchanged when shift is 0" in {
    test(new Activation(width, activations, 0)) { dut =>
      check(dut, 0, 1)      // 1
      check(dut, 0, 5)      // 5
      check(dut, 0, actMax) // 127, largest value that fits
    }
  }

  it should "divide by 2^shift and round down" in {
    test(new Activation(width, activations, 4)) { dut =>
      check(dut, 4, 15)  // 15 / 16  = 0
      check(dut, 4, 16)  // 16 / 16  = 1
      check(dut, 4, 160) // 160 / 16 = 10
      check(dut, 4, 175) // 175 / 16 = 10.9 -> 10
    }
  }

  it should "clamp sums that are too large to fit" in {
    test(new Activation(width, activations, 0)) { dut =>
      check(dut, 0, actMax + 1) // 128    -> 127
      check(dut, 0, sumMax)     // 131071 -> 127
    }
    test(new Activation(width, activations, 4)) { dut =>
      check(dut, 4, 2047)   // 2047 >> 4 = 127, exactly at the limit
      check(dut, 4, 2048)   // 2048 >> 4 = 128 -> 127
      check(dut, 4, sumMax) // 131071 >> 4 = 8191 -> 127
    }
  }

  it should "match a software model on random inputs" in {
    val rnd = new Random(42)
    def r() = rnd.nextInt(sumMax - sumMin + 1) + sumMin
    for (shift <- Seq(0, 4, 8, 10)) {
      test(new Activation(width, activations, shift)) { dut =>
        for (_ <- 0 until 100) {
          check(dut, shift, r())
        }
      }
    }
  }
}