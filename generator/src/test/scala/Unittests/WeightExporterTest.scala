package service

import java.nio.file.Files
import model.{LayerParams, NetworkModel, QuantizationConfig}
import org.scalatest.flatspec.AnyFlatSpec

class WeightExporterTest extends AnyFlatSpec {
    "WeightExporter" should "preserve Scala parameters and export widths for multiple configurations" in {
        val directory = Files.createTempDirectory("quantized-weights-")
        try {
            for (bits <- Seq(4, 8, 12)) {
                val config = QuantizationConfig(weightBits = bits, activationBits = bits)
                val network = WeightLoader.load("../python/models/csv", config)
                WeightExporter.write(network, directory.toString)
                for (layer <- network.layers) {
                    val weights = Files.readString(directory.resolve(s"${layer.index}_weight.csv"))
                        .linesIterator.map(_.split(",").map(_.toInt).toSeq).toSeq
                    val bias = Files.readString(directory.resolve(s"${layer.index}_bias.csv"))
                        .linesIterator.map(_.toInt).toSeq
                    assert(weights == layer.weights)
                    assert(bias == layer.bias)
                }
                val metadata = Files.readString(directory.resolve("network.json"))
                assert(metadata.contains(s"\"weight_bits\": $bits"))
                assert(metadata.contains(s"\"activation_bits\": $bits"))
                assert(metadata.contains(s"\"accumulator_bits\":${2 * bits + 10}"))
                assert(metadata.contains("\"activation\":\"identity\""))
                if (bits == 8) assert(network.layers.map(_.shift) == Seq(13, 11))
            }
            Files.writeString(directory.resolve("test_inputs.csv"), "stale\n")
            WeightExporter.write(WeightLoader.load("../python/models/csv"), directory.toString)
            assert(!Files.exists(directory.resolve("test_inputs.csv")))
            val overflowing = LayerParams(0, Seq(Seq(127)), Seq(Int.MaxValue), 0)
            intercept[IllegalArgumentException](WeightExporter.write(NetworkModel(Seq(overflowing)), directory.toString))
            intercept[IllegalArgumentException](WeightLoader.load("../python/models/csv", QuantizationConfig(biasBits = 4)))
            intercept[IllegalArgumentException](QuantizationConfig(weightBits = 32))
        } finally {
            val files = Files.list(directory)
            try files.forEach(path => Files.delete(path)) finally files.close()
            Files.delete(directory)
        }
    }

    it should "use the configured weight and activation widths independently" in {
        assert(WeightLoader.quantizeWeight(0.5, 4) == 4)
        assert(WeightLoader.quantizeWeight(1.0, 4) == 7)
        assert(WeightLoader.quantizeWeight(-1.0, 4) == -8)
        assert(WeightLoader.activationShift(Seq(Seq(1, -2), Seq(3, 4)), Seq(10, -5), 4) == 3)
        val config = QuantizationConfig(weightBits = 4, activationBits = 8, biasBits = 16)
        val network = WeightLoader.load("../python/models/csv", config)
        assert(network.quantization == config)
        assert(network.layers.head.weights.flatten.forall(w => w >= -8 && w <= 7))
    }
}
