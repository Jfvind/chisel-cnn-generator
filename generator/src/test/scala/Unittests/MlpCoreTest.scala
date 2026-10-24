package mlp

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

import model.{LayerParams, NetworkModel}
import service.WeightLoader
import shell.AccelConfig

class MlpCoreTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "MlpCore"

    // Software model: hidden layers ReLU -> >> shift -> clamp 255, last layer raw sum
    def reference(net: NetworkModel, x: Seq[Int]): Seq[Long] =
        net.layers.zipWithIndex.foldLeft(x.map(_.toLong)) { case (in, (l, k)) =>
            val sums = (0 until l.outFeatures).map(j => in.zip(l.weights).map { case (a, row) => a * row(j) }.sum + l.bias(j))
            if (k == net.layers.length - 1) sums else sums.map(s => math.min(math.max(s, 0L) >> l.shift, 255L))
        }

    // Hidden activations that saturate (what OVF_COUNT must report for one input)
    def overflows(net: NetworkModel, x: Seq[Int]): Int =
        net.layers.init.foldLeft((x.map(_.toLong), 0)) { case ((in, n), l) =>
            val shifted = (0 until l.outFeatures).map(j =>
                math.max(in.zip(l.weights).map { case (a, row) => a * row(j) }.sum + l.bias(j), 0L) >> l.shift)
            (shifted.map(math.min(_, 255L)), n + shifted.count(_ > 255L))
        }._2

    def argmax(xs: Seq[Long]): Int = xs.indexOf(xs.max)

    def cfgFor(net: NetworkModel) =
        AccelConfig(core = "mlp", nIn = net.layers.head.inFeatures, nOut = net.layers.last.outFeatures, streamW = 32)

    // Fixed cycles at full speed: sum_l (in_l + 2) + nOut - 1 (see MlpCore header)
    def fullSpeedCycles(net: NetworkModel): Int = net.layers.map(_.inFeatures + 2).sum + net.layers.last.outFeatures - 1

    case class Run(out: Seq[Long], lasts: Seq[Int], dones: Int, firstIn: Int, lastOut: Int)

    // One input vector in, nOut beats out. stall = Some(rnd): random VALID/READY, None: full speed.
    def stream(dut: MlpCore, x: Seq[Int], nOut: Int, stall: Option[Random]): Run = {
        val got = scala.collection.mutable.ArrayBuffer[Long]()
        val lasts = scala.collection.mutable.ArrayBuffer[Int]()
        var sent = 0; var holding = false; var cyc = 0; var dones = 0; var firstIn = -1; var lastOut = -1
        while (got.length < nOut) {
            if (!holding && sent < x.length && stall.forall(_.nextInt(4) != 0)) holding = true
            dut.io.in.valid.poke(holding.B)
            if (sent < x.length) { dut.io.in.bits.data.poke(x(sent).U); dut.io.in.bits.last.poke((sent == x.length - 1).B) }
            val ready = stall.forall(_.nextInt(3) != 0)
            dut.io.out.ready.poke(ready.B)
            val inFire = holding && dut.io.in.ready.peek().litToBoolean
            if (inFire && firstIn < 0) firstIn = cyc
            if (ready && dut.io.out.valid.peek().litToBoolean) {
                if (dut.io.out.bits.last.peek().litToBoolean) { lasts += got.length; lastOut = cyc }
                val raw = dut.io.out.bits.data.peek().litValue
                got += (if (raw >= (BigInt(1) << 31)) raw - (BigInt(1) << 32) else raw).toLong   // two's complement
            }
            if (dut.io.done.peek().litToBoolean) dones += 1
            dut.clock.step()
            if (inFire) { sent += 1; holding = false }
            cyc += 1
            assert(cyc < 20 * (x.length + 50) + 1000, "stream timeout")
        }
        dut.io.in.valid.poke(false.B); dut.io.out.ready.poke(false.B)
        Run(got.toSeq, lasts.toSeq, dones, firstIn, lastOut)
    }

    // Logits, TLAST, done, RESULT and idle after the input; at full speed also the documented cycle count
    def checkImage(dut: MlpCore, net: NetworkModel, x: Seq[Int], stall: Option[Random]): Run = {
        val want = reference(net, x)
        dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)   // CTRL start: clears OVF_COUNT
        val r = stream(dut, x, want.length, stall)
        assert(r.out == want, s"logits for input $x")
        assert(r.lasts == Seq(want.length - 1), s"TLAST must be on the last beat only, got ${r.lasts}")
        assert(r.dones == 1, "done must pulse once per input")
        dut.io.result.expect(argmax(want).U)
        dut.io.busy.expect(false.B, "must be idle after the last beat")
        dut.io.ovf.expect(overflows(net, x).U, "OVF_COUNT = saturated hidden activations of this input")
        if (stall.isEmpty) assert(r.lastOut - r.firstIn == fullSpeedCycles(net), "cycles at full speed")
        r
    }

    // 3 -> 2 -> 2, worked out by hand
    //   layer 0: w = [[2, -1], [1, 3], [-4, 2]], b = [10, -20], shift 1 (hidden: ReLU, >> 1, clamp 255)
    //   layer 1: w = [[1, -2], [1, 3]],          b = [0, 5]           (last: raw sum)
    val tiny = NetworkModel(Seq(
        LayerParams(0, Seq(Seq(2, -1), Seq(1, 3), Seq(-4, 2)), Seq(10, -20), 1),
        LayerParams(1, Seq(Seq(1, -2), Seq(1, 3)), Seq(0, 5), 0)
    ))

    it should "compute a tiny hand-checked network" in {
        test(new MlpCore(cfgFor(tiny), tiny)) { dut =>
            // x = [100, 50, 20]: h0 = 200 + 50 - 80 + 10 = 180 >> 1 = 90, h1 = -100 + 150 + 40 - 20 = 70 >> 1 = 35
            //   o0 = 90 + 35 = 125, o1 = -180 + 105 + 5 = -70 -> class 0
            assert(checkImage(dut, tiny, Seq(100, 50, 20), None).out == Seq(125L, -70L))
            // x = [255, 255, 0]: h0 = 510 + 255 + 10 = 775 >> 1 = 387 -> clamps to 255, h1 = -255 + 765 - 20 = 490 >> 1 = 245
            //   o0 = 255 + 245 = 500, o1 = -510 + 735 + 5 = 230 -> class 0
            assert(checkImage(dut, tiny, Seq(255, 255, 0), None).out == Seq(500L, 230L))
            // x = [0, 0, 100]: h0 = -400 + 10 < 0 -> ReLU 0, h1 = 200 - 20 = 180 >> 1 = 90
            //   o0 = 0 + 90 = 90, o1 = 0 + 270 + 5 = 275 -> class 1
            assert(checkImage(dut, tiny, Seq(0, 0, 100), None).out == Seq(90L, 275L))
            dut.io.result.expect(1.U)
        }
    }

    // Random network of random shape: 1-3 layers, 1-16 neurons each.
    // The shift is one below WeightLoader's choice so some activations clamp.
    def randomNet(rnd: Random): NetworkModel = {
        val sizes = Seq.fill(rnd.nextInt(3) + 2)(rnd.nextInt(16) + 1)
        NetworkModel(sizes.zip(sizes.tail).zipWithIndex.map { case ((in, out), k) =>
            val w = Seq.fill(in)(Seq.fill(out)(rnd.nextInt(256) - 128))
            val b = Seq.fill(out)(rnd.nextInt(20001) - 10000)
            LayerParams(k, w, b, (WeightLoader.activationShift(w, b) - 1) max 0)
        })
    }

    // New seed every run; rerun a failure with: sbt -Dseed=<seed> "testOnly mlp.MlpCoreTest"
    val seed = sys.props.get("seed").map(_.toLong).getOrElse(System.nanoTime())

    it should s"match the reference model on random networks, with and without backpressure (seed $seed)" in {
        val rnd = new Random(seed)
        for (_ <- 0 until 5) {
            val net   = randomNet(rnd)
            val shape = (net.layers.head.inFeatures +: net.layers.map(_.outFeatures)).mkString(" -> ")
            withClue(s"network $shape, seed $seed: ") {
                test(new MlpCore(cfgFor(net), net)) { dut =>
                    def input() = Seq.fill(net.layers.head.inFeatures)(rnd.nextInt(256))
                    for (_ <- 0 until 6) checkImage(dut, net, input(), Some(rnd))  // random stalls, back to back
                    for (_ <- 0 until 2) checkImage(dut, net, input(), None)       // full speed: cycle count
                }
            }
        }
    }

    it should "resolve nIn, nOut and the weight hash from the network" in {
        val dir = new java.io.File(getClass.getResource("/weights").toURI).getPath
        val cfg = AccelConfig.resolve(AccelConfig(core = "mlp", nIn = 1, nOut = 1, weightBiasDir = dir))
        assert(cfg.nIn == 784 && cfg.nOut == 10 && cfg.paramHash != 0)
        val typical = AccelConfig.resolve(cfg.copy(shiftPolicy = "typical"))
        assert(typical.cfgHash != cfg.cfgHash, "other shifts -> other weights in hardware -> other CFG_HASH")
    }

    it should "elaborate the trained MNIST network" in {
        val dir = new java.io.File(getClass.getResource("/weights").toURI).getPath
        val net = WeightLoader.load(dir)
        val sv  = circt.stage.ChiselStage.emitSystemVerilog(new MlpCore(AccelConfig(core = "mlp", streamW = 8), net))
        assert("\\$readmemh".r.findAllIn(sv).length == 128, "one initialised weight bank per lane")
    }
}
