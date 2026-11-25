package mlp

import chisel3._
import chisel3.util._
import chisel3.util.experimental.loadMemoryFromFileInline
import java.io.File
import java.nio.file.Files

import model.NetworkModel

// One weight memory per lane, hardcoded from the network. All lanes read the same row each clock.
// Row layerBase(l) + i holds w_l[Ki + j][n] in bits j*bits, j < k, for lane n; layerBase(l) = sum of the
// rows of the layers before l (rows per layer = groups). 0 when lane n is unused in layer l or Ki + j is past
// the layer's inputs. MNIST 784 -> 128 -> 10, k = 4: 228 rows x 32 bits per lane.
// Latency 2: memory read (1 cycle) + output register (1 cycle).
class WeightBanks(net: NetworkModel, lanes: Int, k: Int, bits: Int, groups: Seq[Int]) extends Module {
    val depth = groups.sum
    val io = IO(new Bundle {
        val addr = Input(UInt(log2Ceil(depth max 2).W))
        val en   = Input(Bool())
        val w    = Output(Vec(lanes, Vec(k, SInt(bits.W))))     // w(n) = lane n's k weights of the row
    })

    val hexDir = new File(f"generated/mlp-${net.hashCode}%08x").getAbsoluteFile   // absolute: Vivado's $readmemh must find it
    hexDir.mkdirs()
    for (n <- 0 until lanes) {
        val mem = SyncReadMem(depth, UInt((k * bits).W))        // stored as raw bits; a SInt memory loads "fe" as 254 in treadle
        // forward slashes: a Windows path "C:\Users\..." inside a Verilog string would be read as escape sequences
        loadMemoryFromFileInline(mem, writeBankHex(n).getAbsolutePath.replace('\\', '/'))
        val rowReg = RegNext(mem.read(io.addr, io.en))
        io.w(n) := VecInit(MlpCore.fields(rowReg, k, bits).map(_.asSInt))
    }

    // Writes lane n's weights as bank_<n>.hex: one row per k-input group, weight j in bits j*bits,
    // two's complement per weight (-2 @ 8 bit -> "fe")
    def writeBankHex(n: Int): File = {
        val rows = net.layers.zip(groups).flatMap { case (l, g) =>
            l.weights.map(row => if (n < l.outFeatures) row(n) else 0).padTo(g * k, 0).grouped(k)
        }
        val digits = (k * bits + 3) / 4
        val lines = rows.map { r =>
            val word = r.zipWithIndex.map { case (w, j) => BigInt(w & ((1 << bits) - 1)) << (j * bits) }.sum
            String.format(s"%0${digits}x", word.bigInteger)
        }
        val file = new File(hexDir, s"bank_$n.hex")
        Files.write(file.toPath, lines.mkString("", "\n", "\n").getBytes)
        file
    }
}
