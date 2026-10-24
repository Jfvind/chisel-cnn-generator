package service

import java.io.File
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec
import model.QuantizationConfig


class WeightLoaderTest extends AnyFlatSpec {
    behavior of "WeightLoader"

    val dir     = new File(getClass.getResource("/weights").toURI)
    val typical = QuantizationConfig(shiftPolicy = "typical")
    val network = WeightLoader.load(dir.getPath, typical)   // the expectations below are for the typical policy

    def fileLines(name: String) = Source.fromFile(new File(dir, name)).getLines().toSeq

    it should "load the layers in index order with the correct shapes" in {
        assert(network.layers.map(_.index) == Seq(0, 2))
        assert(network.layers.map(l => (l.inFeatures, l.outFeatures)) == Seq((784, 128), (128, 10)))
    }

    it should "quantize weights with a fixed scale and reject values outside [-1, 1]" in {
        assert(WeightLoader.quantizeWeight(0.5) == 64)
        assert(WeightLoader.quantizeWeight(-1.0) == -128)
        assert(WeightLoader.quantizeWeight(1.0) == 127)
        assertThrows[IllegalArgumentException](WeightLoader.quantizeWeight(1.5))
    }

    it should "compute the activation shift from the typical sum, not the worst case" in {
        // 16 inputs, weights +10/-10: mean 127.5 * 0 = 0, std 255/sqrt(12) * sqrt(16 * 100) = 2944.5
        // typical = 3 * 2944.5 = 8833: >> 5 = 276 > 255, >> 6 = 138 fits -> shift 6
        // (the worst case 255 * 8 * 10 = 20400 would need shift 7)
        val alternating = Seq.tabulate(16)(i => Seq(if (i % 2 == 0) 10 else -10))
        assert(WeightLoader.activationShift(alternating, Seq(0), typical) == 6)
        assert(WeightLoader.activationShift(alternating, Seq(0)) == 7)            // default policy: worst case
        // Neuron 1 of [[1, -2], [3, 4]]: 127.5 * 2 - 5 + 3 * 73.6 * sqrt(20) = 1237.6 -> >> 3 = 154.7 fits
        assert(WeightLoader.activationShift(Seq(Seq(1, -2), Seq(3, 4)), Seq(10, -5), typical) == 3)
        assert(WeightLoader.activationShift(Seq(Seq(0)), Seq(0), typical) == 0) // nothing to shift
    }

    it should "compute the activation shift for each layer" in {
        // Typical sums: 344124 (layer 0) and 46266 (layer 2)
        assert(network.layers.map(_.shift) == Seq(11, 8))
        assert(WeightLoader.load(dir.getPath).layers.map(_.shift) == Seq(13, 11))   // default policy: worst case
    }

    it should "keep weights as [in][out] in file order" in {
        val Seq(l0, l2) = network.layers
        assert(l0.weights(0)(0) == 2)     //  0.0154 * 128
        assert(l0.weights(1)(0) == -2)    // -0.0187 * 128
        assert(l0.weights(783)(127) == 0) // -0.00002 * 128

        // Row 0 col 1 vs row 1 col 0 differ, so a transpose or row swap would fail here
        assert(l2.weights(0)(0) == 24)    //  0.1875 * 128
        assert(l2.weights(0)(1) == 6)     //  0.0483 * 128
        assert(l2.weights(1)(0) == -1)    // -0.0112 * 128
        assert(l2.weights(127)(9) == -61) // -0.4753 * 128
    }

    // Bias sits at actScale * 128: layer 0 inputs are pixels (255), layer 2 inputs are layer 0 acts (255 * 128 / 2^shift0)
    val biasScale = Map(0 -> 255.0 * 128, 2 -> 255.0 * 128 / (1 << network.layers(0).shift) * 128)

    it should "keep bias in file order at the act * weight scale" in {
        val Seq(l0, l2) = network.layers
        assert(l0.bias.take(2) == Seq(4464, 1218)) // 0.1368 * 32640, 0.0373 * 32640
        assert(l0.bias.last == 3136)
        assert(l2.bias.take(2) == Seq(-126, -48))  // -0.0620 * 2040, -0.0237 * 2040
        assert(l2.bias.last == 31)
    }

    it should "match every value of every file, line by line" in {
        for (l <- network.layers) {
            val weightRows = fileLines(s"${l.index}_weight.csv").map(_.split(",").map(x => math.round(x.toDouble * 128).toInt).toSeq)
            val bias       = fileLines(s"${l.index}_bias.csv").map(x => math.round(x.toDouble * biasScale(l.index)).toInt)
            assert(l.weights == weightRows)
            assert(l.bias == bias)
            // Bias j belongs to weight column j (output neuron j)
            assert(l.weights.forall(_.length == l.bias.length))
        }
    }
}
