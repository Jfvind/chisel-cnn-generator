package model

case class QuantizationConfig(weightBits: Int = 8, activationBits: Int = 8, biasBits: Int = 32) {
    require(weightBits >= 2 && weightBits <= 16, "Weight width must be 2..16 bits")
    require(activationBits >= 1 && activationBits <= 16, "Activation width must be 1..16 bits")
    require(biasBits >= 2 && biasBits <= 32, "Bias width must be 2..32 bits")

    val weightScale: Int = 1 << (weightBits - 1)
    val activationMax: Int = (1 << activationBits) - 1

    def accumulatorBits(inputs: Int): Int = {
        require(inputs > 0, "Layers must have at least one input")
        val bits = weightBits + activationBits + chisel3.util.log2Ceil(inputs)
        require(bits <= 63, "Accumulator width must fit signed int64 reference arithmetic")
        bits
    }
}
