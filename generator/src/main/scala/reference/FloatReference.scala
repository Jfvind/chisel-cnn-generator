package reference

import service.WeightLoader.FloatLayer

/*
The trained network before quantization, run on the float weights train_model.py exported. It gives
the float accuracy that the integer accuracy is compared against, so the cost of quantization can be
measured without loading the PyTorch checkpoint. The network is the one build_model in
train_model.py makes: every layer is sum = inputs . weights + bias, followed by ReLU on all layers but
the last, whose raw scores pick the class (highest score, lowest index on ties).

The arithmetic is done in Double rather than PyTorch's float32, so a score can differ in the last
bits; that only changes a prediction when two classes are almost tied, and the accuracy on the
MNIST test set is the same 96.61 % as PyTorch reports. Inputs are pixels in 0.0..1.0, as torchvision's
ToTensor hands them to the model.
*/

class FloatReference(layers: Seq[FloatLayer]) {
    private class DenseLayer(val inputCount: Int, val outputCount: Int, val weights: Array[Double], val bias: Array[Double])

    private val denseLayers = layers.map(layer =>
        new DenseLayer(layer.weights.length, layer.bias.length, layer.weights.flatten.toArray, layer.bias.toArray))

    def predict(pixels: Array[Double]): Int = {
        val lastHidden = denseLayers.init.foldLeft(pixels)((activations, layer) => sums(layer, activations).map(math.max(_, 0.0)))
        val scores = sums(denseLayers.last, lastHidden)
        scores.indices.foldLeft(0)((best, index) => if (scores(index) > scores(best)) index else best)
    }

    private def sums(layer: DenseLayer, activations: Array[Double]): Array[Double] = {
        require(activations.length == layer.inputCount, s"Expected ${layer.inputCount} inputs, got ${activations.length}")
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
        result
    }
}
