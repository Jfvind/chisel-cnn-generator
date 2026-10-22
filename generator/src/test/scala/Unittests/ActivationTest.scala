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
  val actMax      = (1 << width) - 1            //  255

  // Software model: ReLU -> divide by 2^shift (rounding down) -> clamp to actMax
  def model(sum: Int, shift: Int): Int = math.min(math.max(sum, 0) >> shift, actMax)

  def check(dut: Activation, shift: Int, sum: Int): Unit = {
    dut.io.shift.poke(shift.U)
    dut.io.sum.poke(sum.S)
    dut.io.act.expect(model(sum, shift).U)
  }

  it should "output 0 for negative sums and zero" in {
    test(new Activation(width, activations)) { dut =>
      check(dut, 4, 0)      // 0
      check(dut, 4, -1)     // 0
      check(dut, 4, -100)   // 0
      check(dut, 4, sumMin) // most negative input -> 0
    }
  }

  it should "pass small positive sums through unchanged when shift is 0" in {
    test(new Activation(width, activations)) { dut =>
      check(dut, 0, 1)      // 1
      check(dut, 0, 5)      // 5
      check(dut, 0, actMax) // 255, largest value that fits
    }
  }

  it should "divide by 2^shift and round down" in {
    test(new Activation(width, activations)) { dut =>
      check(dut, 4, 15)  // 15 / 16  = 0
      check(dut, 4, 16)  // 16 / 16  = 1
      check(dut, 4, 160) // 160 / 16 = 10
      check(dut, 4, 175) // 175 / 16 = 10.9 -> 10
    }
  }

  it should "clamp to the max value instead of overflowing" in {
    test(new Activation(width, activations)) { dut =>
      check(dut, 0, actMax + 1) // 256 -> 255, would wrap to 0
      check(dut, 0, sumMax)     // largest sum -> 255
      check(dut, 4, 4096)       // 4096 / 16 = 256 -> 255
    }
  }

  it should "change behaviour when the shift input changes" in {
    test(new Activation(width, activations)) { dut =>
      check(dut, 0, 160) // 160
      check(dut, 4, 160) // 10
      check(dut, 8, 160) // 0
    }
  }

  it should "match a software model on random inputs" in {
    val rnd = new Random(42)
    test(new Activation(width, activations)) { dut =>
      for (_ <- 0 until 400) {
        val shift = rnd.nextInt(sumMaxSize)
        check(dut, shift, rnd.nextInt(sumMax - sumMin + 1) + sumMin)
      }
    }
  }
}
