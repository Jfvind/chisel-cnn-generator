package mac

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class MacArrayTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "MacArray"

    val width  = 8
    val lanes  = 4
    val accW   = 26
    val wMin   = -(1 << (width - 1))
    val wMax   = (1 << (width - 1)) - 1
    val actMax = (1 << width) - 1

    def step(dut: MacArray, a: Int, ws: Seq[Int], en: Boolean = true, clear: Boolean = false): Unit = {
        dut.io.a(0).poke(a.U)
        for (n <- 0 until lanes) dut.io.w(n)(0).poke(ws(n).S)
        dut.io.en.poke(en.B); dut.io.clear.poke(clear.B)
        dut.clock.step()
    }

    // One more clock with en low: the products still in the product registers reach acc
    def flush(dut: MacArray): Unit = step(dut, 0, Seq.fill(lanes)(0), en = false)

    def expectAcc(dut: MacArray, model: Seq[Long]): Unit =
        for (n <- 0 until lanes) dut.io.acc(n).expect(model(n).S, s"lane $n")

    it should "broadcast a to every lane, each with its own weight" in {
        test(new MacArray(width, lanes, accW)) { dut =>
            step(dut, 10, Seq(1, 2, -3, 0))
            step(dut, 4,  Seq(5, -5, 1, 7))
            flush(dut)
            expectAcc(dut, Seq(10 + 20, 20 - 20, -30 + 4, 0 + 28))
        }
    }

    it should "apply clear and en to every lane" in {
        test(new MacArray(width, lanes, accW)) { dut =>
            step(dut, 3, Seq(1, 2, 3, 4))
            step(dut, 9, Seq(9, 9, 9, 9), en = false)
            flush(dut)
            expectAcc(dut, Seq(3, 6, 9, 12))
            step(dut, 9, Seq(9, 9, 9, 9), clear = true)
            flush(dut)
            expectAcc(dut, Seq(0, 0, 0, 0))
        }
    }

    it should "match a software model on random inputs" in {
        val rnd = new Random(7)
        test(new MacArray(width, lanes, accW)) { dut =>
            // product register: a product is added one clock later; clear also drops the one in flight
            var model = Seq.fill(lanes)(0L)
            var pend: Option[Seq[Long]] = None
            for (_ <- 0 until 300) {
                val a  = rnd.nextInt(actMax + 1)
                val ws = Seq.fill(lanes)(rnd.nextInt(wMax - wMin + 1) + wMin)
                val en = rnd.nextInt(4) != 0; val clear = rnd.nextInt(25) == 0
                step(dut, a, ws, en, clear)
                model = if (clear) Seq.fill(lanes)(0L) else model.zip(pend.getOrElse(Seq.fill(lanes)(0L))).map { case (m, p) => m + p }
                pend  = if (en && !clear) Some(ws.map(w => a.toLong * w)) else None
                expectAcc(dut, model)
            }
        }
    }
}
