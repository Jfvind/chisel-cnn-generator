package mac

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class MacArrayTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "MacArray"

    val width = 8
    val activations   = 4
    val min   = -(1 << (width - 1))    // -128
    val max   = (1 << (width - 1)) - 1 //  127
    val actMax = (1 << width) - 1      //  255, acts are unsigned (output of ReLU)

    def check(dut: MacArray, acts: Seq[Int], weights: Seq[Int], bias: Int): Unit = {
        for (i <- 0 until activations) {
            dut.io.acts(i).poke(acts(i).U)
            dut.io.weights(i).poke(weights(i).S)
        }
        dut.io.bias.poke(bias.S)
        val expected = acts.zip(weights).map { case (a, w) => a * w }.sum + bias
        dut.io.sum.expect(expected.S)
    }

    it should "compute a simple dot product" in {
        test(new MacArray(width, activations)) { dut =>
            check(dut, Seq(1, 2, 3, 4),   Seq(1, 1, 1, 1), 0) // 10
            check(dut, Seq(1, 2, 3, 4),   Seq(2, -2, 2, -2), 5) // -4 + 5 = 1
        }
    }

    it should "not overflow at extreme values" in {
        test(new MacArray(width, activations)) { dut =>
            check(dut, Seq.fill(activations)(actMax), Seq.fill(activations)(max), max) // largest positive sum
            check(dut, Seq.fill(activations)(actMax), Seq.fill(activations)(min), min) // largest negative sum
        }
    }

    it should "multiply a positive activation with a negative weight correctly" in {
        test(new MacArray(width, activations)) { dut =>
            // Acts >= 128 have the MSB set, so they must not be treated as negative
            check(dut, Seq(200, 0, 0, 0),   Seq(-3, 0, 0, 0),    0)  // 200 * -3 = -600
            check(dut, Seq(actMax, 0, 0, 0), Seq(min, 0, 0, 0),  0)  // 255 * -128 = -32640
            check(dut, Seq(10, 200, 0, 0),  Seq(5, -1, 0, 0),    0)  // 50 - 200 = -150
            check(dut, Seq(150, 100, 0, 0), Seq(-2, 4, 0, 0),    -7) // -300 + 400 - 7 = 93
        }
    }

    it should "match a software model on random inputs" in {
        val rnd = new Random(42)
        def r() = rnd.nextInt(max - min + 1) + min
        def rAct() = rnd.nextInt(actMax + 1)
        test(new MacArray(width, activations)) { dut =>
            for (_ <- 0 until 100) {
                check(dut, Seq.fill(activations)(rAct()), Seq.fill(activations)(r()), r())
            }
        }
    }
}