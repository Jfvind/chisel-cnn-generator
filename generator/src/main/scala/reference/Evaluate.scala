package reference

import java.nio.file.{Files, Path, Paths}
import java.security.MessageDigest

import model.QuantizationConfig
import service.{Csv, QuantizedReader, WeightExporter, WeightLoader}

/*
Evaluates a quantized export on the 10,000 MNIST test images and writes the test vectors the hardware
is checked against. The float network (FloatReference on the trained float CSVs) and the integer
network (IntegerReference on the export) run on the same images; both accuracies are printed and
written to the results CSV, and the first N images are written with their expected integer scores.
`make reference` runs it right after WeightExporter:

    sbt "runMain reference.Evaluate --quantized-dir generated/quantized --float-dir ../python/models/csv"

The vectors are the files in WeightExporter.TestVectorFiles next to the export, in the format of
docs/export-format.md, so chiseltest, python/tests and the board read the same files. parameter_sha256
covers network.json followed by each layer's weight and bias CSV, so vectors that belong to another
export are detected. A pixel enters both networks as torchvision's ToTensor hands it to PyTorch,
float32 byte / 255; the integer network then applies the input rule, and since a byte only has 256
values that is done once into a lookup table.
*/

object Evaluate {
    case class Result(correct: Int, total: Int) { def accuracy: Double = 100.0 * correct / total }

    def pixel(byte: Byte): Double = ((byte & 0xff).toFloat / 255f).toDouble

    def quantize(split: Mnist.Split, quantization: QuantizationConfig): Seq[Seq[Int]] = {
        val table = Array.tabulate(256)(value => quantization.quantizeInput(pixel(value.toByte)))
        split.images.map(image => image.toSeq.map(byte => table(byte & 0xff)))
    }

    def integerAccuracy(reference: IntegerReference, inputs: Seq[Seq[Int]], labels: Seq[Int]): Result =
        Result(inputs.zip(labels).count { case (pixels, label) => reference.predict(pixels) == label }, inputs.length)

    def floatAccuracy(reference: FloatReference, split: Mnist.Split): Result =
        Result(split.images.zip(split.labels).count { case (image, label) => reference.predict(image.map(pixel)) == label }, split.images.length)

    def writeTestVectors(directory: Path, reference: IntegerReference, inputs: Seq[Seq[Int]]): Unit = {
        require(inputs.nonEmpty, "No test vectors to write")
        val Seq(inputsFile, outputsFile, metadataFile) = WeightExporter.TestVectorFiles
        val scores = inputs.map(reference.infer)
        Csv.write(directory.resolve(inputsFile), inputs)
        Csv.write(directory.resolve(outputsFile), scores)
        Files.writeString(directory.resolve(metadataFile),
            s"""{
               |  "schema_version": 1,
               |  "parameter_sha256": "${parameterDigest(directory, reference)}",
               |  "samples": ${inputs.length},
               |  "input_features": ${inputs.head.length},
               |  "output_features": ${scores.head.length}
               |}
               |""".stripMargin)
    }

    def writeResults(file: Path, float: Result, integer: Result, quantization: QuantizationConfig): Unit = {
        Files.createDirectories(file.toAbsolutePath.getParent)
        Files.writeString(file,
            "float_accuracy_percent,quantized_accuracy_percent,test_samples,weight_bits,activation_bits,bias_bits\n" +
            f"${float.accuracy}%.2f,${integer.accuracy}%.2f,${integer.total},${quantization.weightBits},${quantization.activationBits},${quantization.biasBits}\n")
    }

    private def parameterDigest(directory: Path, reference: IntegerReference): String = {
        val digest = MessageDigest.getInstance("SHA-256")
        digest.update(Files.readAllBytes(directory.resolve("network.json")))
        for (layer <- reference.network.layers; kind <- Seq("weight", "bias"))
            digest.update(Files.readAllBytes(directory.resolve(s"${layer.index}_$kind.csv")))
        digest.digest().map(byte => f"${byte & 0xff}%02x").mkString
    }

    def main(arguments: Array[String]): Unit = {
        val allowed = Set("--quantized-dir", "--float-dir", "--mnist-dir", "--test-vectors", "--results")
        val pairs   = arguments.grouped(2).toSeq
        require(arguments.length % 2 == 0 && pairs.forall(pair => allowed(pair.head)) && pairs.map(_.head).distinct == pairs.map(_.head),
            s"Expected one value each for ${allowed.mkString(", ")}")
        val options     = pairs.map(pair => pair.head -> pair(1)).toMap
        val directory   = Paths.get(options.getOrElse("--quantized-dir", "generated/quantized"))
        val vectorCount = options.getOrElse("--test-vectors", "64").toInt
        require(vectorCount >= 1, "--test-vectors must be positive")

        val reference    = new IntegerReference(QuantizedReader.load(directory))
        val quantization = reference.network.quantization
        val split        = Mnist.test(Paths.get(options.getOrElse("--mnist-dir", "../python/data/MNIST/raw")))
        val inputs       = quantize(split, quantization)
        val integer      = integerAccuracy(reference, inputs, split.labels)
        val float        = floatAccuracy(new FloatReference(WeightLoader.readFloatLayers(options.getOrElse("--float-dir", "../python/models/csv"))), split)
        val vectors      = inputs.take(vectorCount)
        writeTestVectors(directory, reference, vectors)
        writeResults(Paths.get(options.getOrElse("--results", "../python/results/quantization_results.csv")), float, integer, quantization)

        println(s"Widths: weights=${quantization.weightBits}, inputs/activations=${quantization.activationBits}, bias=${quantization.biasBits}")
        println(f"Float accuracy: ${float.accuracy}%.2f%% | Quantized accuracy: ${integer.accuracy}%.2f%%")
        println(f"Difference: ${integer.accuracy - float.accuracy}%+.2f percentage points (${integer.total} images)")
        println(s"Exported ${vectors.length} test vectors to $directory")
    }
}
