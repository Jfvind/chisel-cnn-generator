package model

// shiftPolicy: how WeightLoader picks each hidden layer's shift, "worst" (never clamps) or "typical" (see WeightLoader)
case class QuantizationConfig(weightBits: Int = 8, activationBits: Int = 8, biasBits: Int = 32, shiftPolicy: String = "worst") {
    require(weightBits >= 2 && weightBits <= 16, "Weight width must be 2..16 bits")
    require(activationBits >= 1 && activationBits <= 16, "Activation width must be 1..16 bits")
    require(biasBits >= 2 && biasBits <= 32, "Bias width must be 2..32 bits")
    require(Set("worst", "typical").contains(shiftPolicy), s"Shift policy must be worst or typical, got $shiftPolicy")

    val weightScale: Int = 1 << (weightBits - 1)
    val activationMax: Int = (1 << activationBits) - 1
    val biasLimit: Long = 1L << (biasBits - 1)

    // pixel 0.0..1.0 -> input integer (rule 1 in docs/quantization.md)
    def quantizeInput(value: Double): Int = math.floor(value * activationMax + 0.5).max(0).min(activationMax).toInt

    def accumulatorBits(inputs: Int): Int = {
        require(inputs > 0, "Layers must have at least one input")
        val bits = weightBits + activationBits + chisel3.util.log2Ceil(inputs)
        require(bits <= 63, "Accumulator width must fit signed int64 reference arithmetic")
        bits
    }

    def accumulatorLimit(inputs: Int): Long = 1L << (accumulatorBits(inputs) - 1)
}
