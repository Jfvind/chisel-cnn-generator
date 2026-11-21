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

    // One clock: drive inputs, then step
    def step(dut: Mac, a: Int, w: Int, en: Boolean = true, clear: Boolean = false): Unit = {
        dut.io.a(0).poke(a.U); dut.io.w(0).poke(w.S); dut.io.en.poke(en.B); dut.io.clear.poke(clear.B)
        dut.clock.step()
    }

    // One more clock with en low: the product still in the product register reaches acc
    def flush(dut: Mac): Unit = step(dut, 0, 0, en = false)

    // Software model of the two stages: a product enters `pend` and is added to acc one clock later;
    // clear zeroes acc and drops the product in flight
    class Model {
        var acc = 0L; var pend: Option[Long] = None
        def step(p: Long, en: Boolean, clear: Boolean): Unit = {
            acc  = if (clear) 0L else acc + pend.getOrElse(0L)
            pend = if (en && !clear) Some(p) else None
        }
    }

    it should "accumulate one product per clock, one clock behind" in {
        test(new Mac(width, accW)) { dut =>
            dut.io.acc.expect(0.S)
            step(dut, 10, 2);  dut.io.acc.expect(0.S)   // 10 * 2 is in the product register
            step(dut, 20, -1); dut.io.acc.expect(20.S)  // 10 * 2 added
            step(dut, 5, 3);   dut.io.acc.expect(0.S)   // + 20 * -1
            flush(dut);        dut.io.acc.expect(15.S)  // + 5 * 3
        }
    }

    it should "hold its value when en is low" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, 7, 7)
            step(dut, 100, 100, en = false)
            flush(dut)
            dut.io.acc.expect(49.S)
        }
    }

    it should "clear, let clear win over en and drop the product in flight" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, 50, 50); flush(dut)
            dut.io.acc.expect(2500.S)
            step(dut, 9, 9)                    // in flight when clear comes
            step(dut, 1, 1, clear = true)
            flush(dut)
            dut.io.acc.expect(0.S)
            step(dut, 3, 4); flush(dut)
            dut.io.acc.expect(12.S)
        }
    }

    it should "treat activations >= 128 as positive at the signed extremes" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, actMax, wMin); flush(dut); dut.io.acc.expect((-32640).S) // 255 * -128
            step(dut, 1, 1, en = false, clear = true)
            for (k <- 1 to 784) step(dut, actMax, wMax)            // largest sum of an MNIST neuron
            flush(dut)
            dut.io.acc.expect((784 * 255 * 127).S)
            step(dut, 1, 1, en = false, clear = true)
            for (k <- 1 to 784) step(dut, actMax, wMin)
            flush(dut)
            dut.io.acc.expect((-784 * 255 * 128).S)
        }
    }

    it should "add k products per clock" in {
        val rnd = new Random(3)
        test(new Mac(width, accW, 2)) { dut =>
            val model = new Model
            for (_ <- 0 until 200) {
                val as = Seq.fill(2)(rnd.nextInt(actMax + 1)); val ws = Seq.fill(2)(rnd.nextInt(wMax - wMin + 1) + wMin)
                for (j <- 0 until 2) { dut.io.a(j).poke(as(j).U); dut.io.w(j).poke(ws(j).S) }
                dut.io.en.poke(true.B); dut.io.clear.poke(false.B); dut.clock.step()
                model.step(as.zip(ws).map { case (a, w) => a.toLong * w }.sum, en = true, clear = false)
                dut.io.acc.expect(model.acc.S)
            }
        }
    }

    it should "match a software model on random sequences" in {
        val rnd = new Random(42)
        test(new Mac(width, accW)) { dut =>
            val model = new Model
            for (_ <- 0 until 500) {
                val a = rnd.nextInt(actMax + 1); val w = rnd.nextInt(wMax - wMin + 1) + wMin
                val en = rnd.nextInt(4) != 0; val clear = rnd.nextInt(20) == 0
                step(dut, a, w, en, clear)
                model.step(a.toLong * w, en, clear)
                dut.io.acc.expect(model.acc.S)
            }
        }
    }
}
