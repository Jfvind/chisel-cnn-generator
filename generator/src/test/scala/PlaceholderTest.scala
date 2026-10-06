package generator

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec


class PlaceholderTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "Top"

    it should "Pass a signal from input to output" in {
        test(new Top()) { dut =>
            dut.io.in.poke(1.U)
            dut.io.out.expect(1.U)

            dut.io.in.poke(0.U)
            dut.io.out.expect(0.U)
        }
    }
}