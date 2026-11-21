package reference

import java.nio.file.{Files, Path, Paths}

import model.{LayerParams, NetworkModel, QuantizationConfig}
import org.scalatest.flatspec.AnyFlatSpec
import service.{Csv, QuantizedReader, WeightExporter, WeightLoader}

/*
Tests the reference package and QuantizedReader on the committed export and on small hand-worked
networks. The reference must reproduce the committed vectors in generated/quantized exactly (python/tests
recompute the same vectors with an independent implementation), write vector files byte-identical to
the committed ones, and give 96.61 % float and 96.69 % integer accuracy on MNIST. Paths are relative to
generator/, like in the other tests.

The hand-worked network has 2 inputs, 2 hidden neurons and 1 output at 4 bits, so activationMax is
15. The hidden layer has weights [[7, -8], [7, 0]], bias [1, 0] and shift 1, and the output layer has
weights [[1], [-1]] and bias -3. With inputs (15, 15) the first hidden sum is 211, which shifts to
105 and saturates to 15, while the second is -120 and becomes 0 through ReLU, so the output is
15 - 3 = 12. With inputs (1, 0) the hidden sums are 8 and -8, giving 4 and 0 and an output of 1. With
inputs (0, 0) the first hidden sum is 1, which the shift truncates to 0, so the output is -3 and stays
negative because the last layer has no ReLU.

The overflow network has 1 input with 2-bit weights and 1-bit activations, so its MAC range is
-4..3, and input 1 with weight 1 and bias 3 sums to 4.
*/

class IntegerReferenceTest extends AnyFlatSpec {
    val exportDirectory = Paths.get("generated/quantized")
    val mnistDirectory  = Paths.get("../python/data/MNIST/raw")

    lazy val exportReference = new IntegerReference(QuantizedReader.load(exportDirectory))
    lazy val exportInputs    = readIntegers(exportDirectory.resolve("test_inputs.csv"))

    def readIntegers(file: Path): Seq[Seq[Int]] = Csv.read(file)(_.toInt)

    def withExportCopy(body: Path => Unit): Unit = {
        val temporaryDirectory = Files.createTempDirectory("reference-export-")
        Files.list(exportDirectory).forEach(file => Files.copy(file, temporaryDirectory.resolve(file.getFileName)))
        try body(temporaryDirectory) finally {
            val files = Files.list(temporaryDirectory)
            try files.forEach(file => Files.delete(file)) finally files.close()
            Files.delete(temporaryDirectory)
        }
    }

    behavior of "IntegerReference"

    it should "reproduce Python's expected scores on the checked-in vectors" in {
        val expected = Csv.read(exportDirectory.resolve("expected_outputs.csv"))(_.toLong)
        assert(exportInputs.map(exportReference.infer) == expected)
    }

    it should "give the same network from the float CSVs as from the export" in {
        assert(exportReference.network == WeightLoader.load("../python/models/csv"))
    }

    it should "write vectors byte-identical to Python's" in withExportCopy { copy =>
        Evaluate.writeTestVectors(copy, exportReference, exportInputs)
        for (fileName <- WeightExporter.TestVectorFiles)
            assert(Files.readString(copy.resolve(fileName)) == Files.readString(exportDirectory.resolve(fileName)), s"$fileName differs")
    }

    it should "quantize MNIST like Python and match the documented float and integer accuracy" in {
        val split  = Mnist.test(mnistDirectory)
        val inputs = Evaluate.quantize(split, exportReference.network.quantization)
        assert(inputs.take(exportInputs.length) == exportInputs)
        val integer = Evaluate.integerAccuracy(exportReference, inputs, split.labels)
        assert(integer.total == 10000)
        assert(integer.correct == 9669, s"integer accuracy ${integer.accuracy}% (README: 96.69%)")
        val float = Evaluate.floatAccuracy(new FloatReference(WeightLoader.readFloatLayers("../python/models/csv")), split)
        assert(float.correct == 9661, s"float accuracy ${float.accuracy}% (README: 96.61%)")
    }

    it should "apply ReLU, truncating shift and saturation in hidden layers only" in {
        val config      = QuantizationConfig(weightBits = 4, activationBits = 4)
        val hiddenLayer = LayerParams(0, Seq(Seq(7, -8), Seq(7, 0)), Seq(1, 0), shift = 1)
        val outputLayer = LayerParams(1, Seq(Seq(1), Seq(-1)), Seq(-3), shift = 5)
        val reference   = new IntegerReference(NetworkModel(Seq(hiddenLayer, outputLayer), config))
        assert(reference.infer(Seq(15, 15)) == Seq(12L))
        assert(reference.infer(Seq(1, 0)) == Seq(1L))
        assert(reference.infer(Seq(0, 0)) == Seq(-3L))
        assert(IntegerReference.requantize(-5, 0, 15) == 0)
        assert(IntegerReference.requantize(31, 1, 15) == 15)
        assert(IntegerReference.requantize(29, 1, 15) == 14)
    }

    it should "pick the lowest index on ties and quantize pixels with round-half-up" in {
        assert(IntegerReference.argmax(Seq(3L, 7L, 7L, 1L)) == 1)
        assert(IntegerReference.argmax(Seq(-5L, -5L)) == 0)
        val eightBit = QuantizationConfig()
        assert(QuantizationConfig(activationBits = 1).quantizeInput(0.5) == 1)
        assert(eightBit.quantizeInput(1.2) == 255)
        assert(eightBit.quantizeInput(-0.1) == 0)
        assert((0 to 255).forall(byte => eightBit.quantizeInput((byte.toFloat / 255f).toDouble) == byte))
    }

    it should "reject inputs out of range and accumulator overflow" in {
        intercept[IllegalArgumentException](exportReference.infer(Seq.fill(784)(256)))
        intercept[IllegalArgumentException](exportReference.infer(Seq.fill(10)(0)))
        val config      = QuantizationConfig(weightBits = 2, activationBits = 1)
        val overflowing = new IntegerReference(NetworkModel(Seq(LayerParams(0, Seq(Seq(1)), Seq(3), 0)), config))
        intercept[ArithmeticException](overflowing.infer(Seq(1)))
    }

    behavior of "QuantizedReader"

    it should "reject exports with unsupported policies" in withExportCopy { copy =>
        val json = Files.readString(copy.resolve("network.json"))
        Files.writeString(copy.resolve("network.json"), json.replace("\"saturate\"", "\"wrap\""))
        intercept[IllegalArgumentException](QuantizedReader.load(copy))
    }
}
