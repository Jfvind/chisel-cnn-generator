package service

import java.nio.file.{Files, Paths}
import model.NetworkModel

object WeightExporter {
    def write(network: NetworkModel, directory: String): Unit = {
        require(network.layers.nonEmpty, "Cannot export an empty network")
        val output = Paths.get(directory)
        Files.createDirectories(output)
        for (layer <- network.layers) { 
            val bits = NetworkModel.Bits * 2 + chisel3.util.log2Ceil(layer.inFeatures) //calc n-bit for MAC sum
            for (neuron_index <- layer.bias.indices) {
                val bound = 255L * layer.weights.map(row => math.abs(row(neuron_index).toLong)).sum + math.abs(layer.bias(neuron_index).toLong)
                require(bound < (1L << (bits - 1)), s"Layer ${layer.index} can overflow its MAC width")
            } //255 * sum of wights abs + bias abs
            val weights = layer.weights.map(_.mkString(",")).mkString("\n") + "\n"
            Files.writeString(output.resolve(s"${layer.index}_weight.csv"), weights)
            Files.writeString(output.resolve(s"${layer.index}_bias.csv"), layer.bias.mkString("\n") + "\n")
        }
        val parameters = network.layers.map(layer => s"${layer.index},1,${layer.shift}")
        Files.writeString(output.resolve("layers.csv"), (Seq("index,multiplier,shift") ++ parameters).mkString("\n") + "\n")
    }

    def main(args: Array[String]): Unit = {
        val network = WeightLoader.load("../python/models/csv")
        write(network, "generated/quantized")
        println(network)
    }
}
