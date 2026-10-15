package service

import java.io.File
import scala.io.Source
import org.scalatest.flatspec.AnyFlatSpec


class WeightLoaderTest extends AnyFlatSpec {
    behavior of "WeightLoader"

    val dir     = new File(getClass.getResource("/weights").toURI)
    val network = WeightLoader.load(dir.getPath)

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

    it should "compute the activation shift from the worst case sum" in {
        // Neuron 0: 255 * (1 + 3) + 10 = 1030, neuron 1: 255 * 4 - 5 = 1015 (the -2 sees act 0)
        // 1030 >> 2 = 257 > 255, 1030 >> 3 = 128 fits -> shift 3
        assert(WeightLoader.activationShift(Seq(Seq(1, -2), Seq(3, 4)), Seq(10, -5)) == 3)
        assert(WeightLoader.activationShift(Seq(Seq(1)), Seq(0)) == 0) // 255 fits as is
    }

    it should "compute the activation shift for each layer" in {
        // Worst case sums: 1087978 (layer 0) and 266208 (layer 2)
        assert(network.layers.map(_.shift) == Seq(13, 11))
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

    // Bias sits at actScale * 128: layer 0 inputs are pixels (255), layer 2 inputs are layer 0 acts (255 * 128 / 2^13)
    val biasScale = Map(0 -> 255.0 * 128, 2 -> 255.0 * 128 / (1 << 13) * 128)

    it should "keep bias in file order at the act * weight scale" in {
        val Seq(l0, l2) = network.layers
        assert(l0.bias.take(2) == Seq(4464, 1218)) // 0.1368 * 32640, 0.0373 * 32640
        assert(l0.bias.last == 3136)
        assert(l2.bias.take(2) == Seq(-32, -12))   // -0.0620 * 510, -0.0237 * 510
        assert(l2.bias.last == 8)
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
