package mac

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

// Software model of one accumulator behind Mac.Latency stages: a product enters the pipeline on en and
// is in acc Mac.Latency clocks later; clear zeroes acc and drops every product in flight
class MacModel {
    var acc = 0L
    private var pipe = Seq.fill(Mac.Latency - 1)(Option.empty[Long])
    def step(p: Long, en: Boolean, clear: Boolean): Unit = {
        acc  = if (clear) 0L else acc + pipe.last.getOrElse(0L)
        pipe = if (clear) pipe.map(_ => None) else (if (en) Some(p) else None) +: pipe.init
    }
}

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

    // Clocks with en low until the products in flight have reached acc
    def flush(dut: Mac): Unit = for (_ <- 1 until Mac.Latency) step(dut, 0, 0, en = false)

    it should "accumulate one product per clock, Latency clocks behind" in {
        test(new Mac(width, accW)) { dut =>
            dut.io.acc.expect(0.S)
            step(dut, 10, 2);  dut.io.acc.expect(0.S)   // 10 * 2 in the product register
            step(dut, 20, -1); dut.io.acc.expect(0.S)   // 10 * 2 in the sum register
            step(dut, 5, 3);   dut.io.acc.expect(20.S)  // 10 * 2 added
            flush(dut);        dut.io.acc.expect(15.S)  // + 20 * -1 + 5 * 3
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

    it should "clear, let clear win over en and drop the products in flight" in {
        test(new Mac(width, accW)) { dut =>
            step(dut, 50, 50); flush(dut)
            dut.io.acc.expect(2500.S)
            step(dut, 9, 9); step(dut, 8, 8)    // both in flight when clear comes
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
            val model = new MacModel
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
            val model = new MacModel
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

// Two lanes per multiplier: both accumulators must equal what two separate Macs would give
class MacPairTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "MacPair"

    val accW   = 26
    val actMax = 255

    def drive(dut: MacPair, as: Seq[Int], w0: Seq[Int], w1: Seq[Int], en: Boolean, clear: Boolean): Unit = {
        for (j <- as.indices) { dut.io.a(j).poke(as(j).U); dut.io.w0(j).poke(w0(j).S); dut.io.w1(j).poke(w1(j).S) }
        dut.io.en.poke(en.B); dut.io.clear.poke(clear.B)
        dut.clock.step()
    }

    it should "pack only widths that fit a DSP48E1" in {
        assert(MacPair.fits(8) && MacPair.fits(4) && !MacPair.fits(9))
    }

    it should "unpack both products exactly for every sign and extreme of a, w0 and w1" in {
        val corners = Seq(-128, -127, -1, 0, 1, 127)
        test(new MacPair(8, accW, 1)) { dut =>
            for (a <- Seq(0, 1, 128, 254, 255); w0 <- corners; w1 <- corners) {
                drive(dut, Seq(a), Seq(w0), Seq(w1), en = true, clear = true)    // clear wins: starts from 0
                drive(dut, Seq(a), Seq(w0), Seq(w1), en = true, clear = false)
                for (_ <- 1 until Mac.Latency) drive(dut, Seq(0), Seq(0), Seq(0), en = false, clear = false)
                dut.io.acc0.expect((a * w0).S, s"a=$a w0=$w0 w1=$w1")
                dut.io.acc1.expect((a * w1).S, s"a=$a w0=$w0 w1=$w1")
            }
        }
    }

    it should "reach the largest MNIST sums in both lanes, with opposite signs" in {
        test(new MacPair(8, accW, 4)) { dut =>
            drive(dut, Seq.fill(4)(0), Seq.fill(4)(0), Seq.fill(4)(0), en = false, clear = true)
            for (_ <- 0 until 784 / 4) drive(dut, Seq.fill(4)(actMax), Seq.fill(4)(-128), Seq.fill(4)(127), en = true, clear = false)
            for (_ <- 1 until Mac.Latency) drive(dut, Seq.fill(4)(0), Seq.fill(4)(0), Seq.fill(4)(0), en = false, clear = false)
            dut.io.acc0.expect((-784 * 255 * 128).S)
            dut.io.acc1.expect((784 * 255 * 127).S)
        }
    }

    it should "match two software Macs on random sequences, k = 4" in {
        val rnd = new Random(11)
        test(new MacPair(8, accW, 4)) { dut =>
            val (m0, m1) = (new MacModel, new MacModel)
            for (_ <- 0 until 500) {
                val as = Seq.fill(4)(rnd.nextInt(actMax + 1))
                val w0 = Seq.fill(4)(rnd.nextInt(256) - 128); val w1 = Seq.fill(4)(rnd.nextInt(256) - 128)
                val en = rnd.nextInt(4) != 0; val clear = rnd.nextInt(30) == 0
                drive(dut, as, w0, w1, en, clear)
                m0.step(as.zip(w0).map { case (a, w) => a.toLong * w }.sum, en, clear)
                m1.step(as.zip(w1).map { case (a, w) => a.toLong * w }.sum, en, clear)
                dut.io.acc0.expect(m0.acc.S); dut.io.acc1.expect(m1.acc.S)
            }
        }
    }

    it should "work for narrower weights too (4 bit)" in {
        val rnd = new Random(5)
        test(new MacPair(4, 16, 2)) { dut =>
            val (m0, m1) = (new MacModel, new MacModel)
            for (_ <- 0 until 300) {
                val as = Seq.fill(2)(rnd.nextInt(16))
                val w0 = Seq.fill(2)(rnd.nextInt(16) - 8); val w1 = Seq.fill(2)(rnd.nextInt(16) - 8)
                val en = rnd.nextInt(4) != 0; val clear = rnd.nextInt(30) == 0
                drive(dut, as, w0, w1, en, clear)
                m0.step(as.zip(w0).map { case (a, w) => a.toLong * w }.sum, en, clear)
                m1.step(as.zip(w1).map { case (a, w) => a.toLong * w }.sum, en, clear)
                dut.io.acc0.expect(m0.acc.S); dut.io.acc1.expect(m1.acc.S)
            }
        }
    }
}
