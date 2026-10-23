package mac

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class MacTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "Mac"

    val width  = 8
    val accW   = 26
    val wMin   = -(1 << (width - 1))   // -128
    val wMax   = (1 << (width - 1)) - 1 //  127
    val actMax = (1 << width) - 1      //  255

    // One clock: drive inputs, check the sum seen so far, then step
    def step(dut: Mac, a: Int, w: Int, en: Boolean = true, clear: Boolean = false): Unit = {
        dut.io.a.poke(a.U); dut.io.w.poke(w.S); dut.io.en.poke(en.B); dut.io.clear.poke(clear.B)
        dut.clock.step()
    }

    it should "accumulate one product per clock" in {
        test(new Mac(width, accW)) { dut =>
            dut.io.acc.expect(0.S)
            step(dut, 10, 2);  dut.io.acc.expect(20.S)  // 10 * 2
            step(dut, 20, -1); dut.io.acc.expect(0.S)   // + 20 * -1
            step(dut, 5, 3);   dut.io.acc.expect(15.S)  // + 5 * 3
        }
    }

    it should "hold its value when en is low" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, 7, 7)
            step(dut, 100, 100, en = false)
            dut.io.acc.expect(49.S)
        }
    }

    it should "clear, and let clear win over en" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, 50, 50)
            step(dut, 1, 1, clear = true)
            dut.io.acc.expect(0.S)
            step(dut, 3, 4)
            dut.io.acc.expect(12.S)
        }
    }

    it should "treat activations >= 128 as positive at the signed extremes" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, actMax, wMin); dut.io.acc.expect((-32640).S) // 255 * -128
            step(dut, 1, 1, clear = true)
            for (k <- 1 to 784) step(dut, actMax, wMax)            // largest sum of an MNIST neuron
            dut.io.acc.expect((784 * 255 * 127).S)
            step(dut, 1, 1, clear = true)
            for (k <- 1 to 784) step(dut, actMax, wMin)
            dut.io.acc.expect((-784 * 255 * 128).S)
        }
    }

    it should "match a software model on random sequences" in {
        val rnd = new Random(42)
        test(new Mac(width, accW)) { dut =>
            var model = 0L
            for (_ <- 0 until 500) {
                val a = rnd.nextInt(actMax + 1); val w = rnd.nextInt(wMax - wMin + 1) + wMin
                val en = rnd.nextInt(4) != 0; val clear = rnd.nextInt(20) == 0
                step(dut, a, w, en, clear)
                model = if (clear) 0 else if (en) model + a * w else model
                dut.io.acc.expect(model.S)
            }
        }
    }
}
