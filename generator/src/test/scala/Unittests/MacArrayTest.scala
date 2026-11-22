package mac

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class MacArrayTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "MacArray"

    val width  = 8
    val accW   = 26
    val wMin   = -(1 << (width - 1))
    val wMax   = (1 << (width - 1)) - 1
    val actMax = (1 << width) - 1

    def step(dut: MacArray, a: Int, ws: Seq[Int], en: Boolean = true, clear: Boolean = false): Unit = {
        dut.io.a(0).poke(a.U)
        for (n <- ws.indices) dut.io.w(n)(0).poke(ws(n).S)
        dut.io.en.poke(en.B); dut.io.clear.poke(clear.B)
        dut.clock.step()
    }

    // Clocks with en low until the products in flight have reached acc
    def flush(dut: MacArray, lanes: Int): Unit = for (_ <- 1 until Mac.Latency) step(dut, 0, Seq.fill(lanes)(0), en = false)

    def expectAcc(dut: MacArray, model: Seq[Long]): Unit =
        for (n <- model.indices) dut.io.acc(n).expect(model(n).S, s"lane $n")

    // Every test runs unpacked (Mac per lane) and packed (MacPair per two lanes); 5 lanes = an odd last lane
    for (pack <- Seq(false, true); lanes <- Seq(4, 5)) {
        val tag = s"(pack = $pack, $lanes lanes)"

        it should s"broadcast a to every lane, each with its own weight $tag" in {
            test(new MacArray(width, lanes, accW, 1, pack)) { dut =>
                val w1 = Seq(1, 2, -3, 0, -128).take(lanes); val w2 = Seq(5, -5, 1, 7, 127).take(lanes)
                step(dut, 10, w1)
                step(dut, 4,  w2)
                flush(dut, lanes)
                expectAcc(dut, w1.zip(w2).map { case (x, y) => 10L * x + 4L * y })
            }
        }

        it should s"apply clear and en to every lane $tag" in {
            test(new MacArray(width, lanes, accW, 1, pack)) { dut =>
                val ws = (1 to lanes)
                step(dut, 3, ws)
                step(dut, 9, Seq.fill(lanes)(9), en = false)
                flush(dut, lanes)
                expectAcc(dut, ws.map(3L * _))
                step(dut, 9, Seq.fill(lanes)(9), clear = true)
                flush(dut, lanes)
                expectAcc(dut, Seq.fill(lanes)(0L))
            }
        }

        it should s"match a software model on random inputs $tag" in {
            val rnd = new Random(7)
            test(new MacArray(width, lanes, accW, 1, pack)) { dut =>
                val models = Seq.fill(lanes)(new MacModel)
                for (_ <- 0 until 300) {
                    val a  = rnd.nextInt(actMax + 1)
                    val ws = Seq.fill(lanes)(rnd.nextInt(wMax - wMin + 1) + wMin)
                    val en = rnd.nextInt(4) != 0; val clear = rnd.nextInt(25) == 0
                    step(dut, a, ws, en, clear)
                    models.zip(ws).foreach { case (m, w) => m.step(a.toLong * w, en, clear) }
                    expectAcc(dut, models.map(_.acc))
                }
            }
        }
    }
}
