package reference

import model.NetworkModel

/*
The bit-exact software model of the quantized network. infer() turns one image of unsigned input
integers into the last layer's wide signed scores, and predict() picks the class from them. The
Chisel tests, the board test and the test vectors all need one trusted answer per image, and keeping
that answer in Scala next to the quantization in WeightLoader puts the whole numerical contract in
one language (issue #60).

The contract is docs/quantization.md, and the arithmetic is the same as MacArray + Activation in
MlpCore and as python/scripts/integer_reference.py:

    hidden layer   sum = inputs . weights + bias,   output = min(max(sum, 0) >> shift, activationMax)
    last layer     sum = inputs . weights + bias,   output = sum (no ReLU, no shift)

Every sum must fit the layer's signed MAC width, and a sum outside it throws, as the contract's
accumulator_overflow = "reject" demands. The weights are copied once into flat arrays in the CSV's
[input][output] order, and inputs that are 0 are skipped because most MNIST pixels are, so 10,000
images take under a second. predict() takes the highest score and the lowest index on ties, like the
RESULT register. Turning pixels into input integers is part of the quantization itself and lives in
QuantizationConfig.quantizeInput.
*/

class IntegerReference(val network: NetworkModel) {
    private val activationMax = network.quantization.activationMax.toLong

    private class DenseLayer(val inputCount: Int, val outputCount: Int, val weights: Array[Long],
                             val bias: Array[Long], val shift: Int, val accumulatorLimit: Long)

    private val layers = network.layers.map { layer =>
        new DenseLayer(layer.inFeatures, layer.outFeatures, layer.weights.flatten.map(_.toLong).toArray,
            layer.bias.map(_.toLong).toArray, layer.shift, network.quantization.accumulatorLimit(layer.inFeatures))
    }

    private val inputCount = layers.head.inputCount

    def infer(pixels: Seq[Int]): Seq[Long] = {
        require(pixels.length == inputCount, s"Expected $inputCount inputs, got ${pixels.length}")
        require(pixels.forall(pixel => pixel >= 0 && pixel <= activationMax), s"Inputs must be in 0..$activationMax")
        val lastHidden = layers.init.foldLeft(pixels.map(_.toLong).toArray) { (activations, layer) =>
            sums(layer, activations).map(sum => IntegerReference.requantize(sum, layer.shift, activationMax))
        }
        sums(layers.last, lastHidden).toSeq
    }

    def predict(pixels: Seq[Int]): Int = IntegerReference.argmax(infer(pixels))

    private def sums(layer: DenseLayer, activations: Array[Long]): Array[Long] = {
        val result = layer.bias.clone()
        var input = 0
        while (input < layer.inputCount) {
            val activation = activations(input)
            if (activation != 0) {
                val row = input * layer.outputCount
                var output = 0
                while (output < layer.outputCount) { result(output) += activation * layer.weights(row + output); output += 1 }
            }
            input += 1
        }
        for (sum <- result if sum < -layer.accumulatorLimit || sum >= layer.accumulatorLimit)
            throw new ArithmeticException(s"MAC accumulator overflow: $sum does not fit ±${layer.accumulatorLimit}")
        result
    }
}

object IntegerReference {
    def requantize(sum: Long, shift: Int, activationMax: Long): Long = math.min(math.max(sum, 0L) >> shift, activationMax)

    def argmax(scores: Seq[Long]): Int = scores.indices.foldLeft(0)((best, index) => if (scores(index) > scores(best)) index else best)
}
