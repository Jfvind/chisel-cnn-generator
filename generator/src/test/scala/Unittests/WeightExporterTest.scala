package service

import java.nio.file.Files
import model.{LayerParams, NetworkModel}
import org.scalatest.flatspec.AnyFlatSpec

class WeightExporterTest extends AnyFlatSpec {
    "WeightExporter" should "preserve every integer produced by the CSV loader" in {
        val network = WeightLoader.load("../python/models/csv")
        val directory = Files.createTempDirectory("quantized-weights-")
        try {
            WeightExporter.write(network, directory.toString)
            for (layer <- network.layers) {
                val weights = Files.readString(directory.resolve(s"${layer.index}_weight.csv"))
                    .linesIterator.map(_.split(",").map(_.toInt).toSeq).toSeq
                val bias = Files.readString(directory.resolve(s"${layer.index}_bias.csv"))
                    .linesIterator.map(_.toInt).toSeq
                assert(weights == layer.weights)
                assert(bias == layer.bias)
            }
            assert(Files.readString(directory.resolve("layers.csv")) == "index,multiplier,shift\n0,1,13\n2,1,11\n")
            val overflowing = LayerParams(0, Seq(Seq(127)), Seq(Int.MaxValue), 0)
            intercept[IllegalArgumentException](WeightExporter.write(NetworkModel(Seq(overflowing)), directory.toString))
        } finally {
            val files = Files.list(directory)
            try files.forEach(path => Files.delete(path)) finally files.close()
            Files.delete(directory)
        }
    }
}
