package shell

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

// =============================================================================
// AccelShellTest.scala — chiseltest tests of AccelShellInner with the stub core.
//
// AxiDriver       : bus-functional model. write()/read() on AXI-Lite and stream() on
//                   AXI-Stream, with random delays on every VALID/READY to catch
//                   ordering and hold-until-handshake bugs.
// AccelShellTest  : ID/config/adder via AXI-Lite, WSTRB byte enables, streaming with
//                   TLAST + backpressure, and a check of the generated top-level port names.
// =============================================================================

// Software model of PS + DMA, simulates hardware. With random delays.
class AxiDriver(dut: AccelShellInner, rnd: Random) {
    private val axi = dut.io.s_axi
    private val maxWait = 100           // cycles until a hang is declared

    def idle(): Unit = {                // All ready/valid signals off at start
        axi.awvalid.poke(false.B); axi.wvalid.poke(false.B); axi.bready.poke(false.B)
        axi.arvalid.poke(false.B); axi.rready.poke(false.B)
        axi.awprot.poke(0.U); axi.arprot.poke(0.U)
        dut.io.s_axis.tvalid.poke(false.B); dut.io.s_axis.tlast.poke(false.B)
        dut.io.m_axis.tready.poke(false.B)
    }

    def write(addr: Int, data: Long, strb: Int = 0xF): Unit = {
        val awDelay = rnd.nextInt(4); val wDelay = rnd.nextInt(4)   // AW and W start at random times
        var awDone = false; var wDone = false; var cyc = 0
        axi.bready.poke(false.B)
        while(!(awDone && wDone)) {
            val driveAw = !awDone && cyc >= awDelay                 // stays high until handshake, per AXI
            val driveW  = !wDone && cyc >= wDelay
            axi.awvalid.poke(driveAw.B); axi.awaddr.poke(addr.U)
            axi.wvalid.poke(driveW.B); axi.wdata.poke(data.U(32.W)); axi.wstrb.poke(strb.U)
            // Handshake
            val awFire = driveAw && axi.awready.peek().litToBoolean
            val wFire  = driveW && axi.wready.peek().litToBoolean
            dut.clock.step()
            if(awFire) awDone = true
            if(wFire) wDone = true
            cyc += 1
            assert(cyc < maxWait, s"write 0x${addr.toHexString}: AW/W handshake timeout")
        }
        axi.awvalid.poke(false.B); axi.wvalid.poke(false.B)
        var n = 0
        while(!axi.bvalid.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < maxWait, "no BVALID") }
        // keeping BREADY low for a few cycles to ensure BVALID stays high
        for(_ <- 0 until rnd.nextInt(3)) { axi.bvalid.expect(true.B, "BVALID dropped before BREADY"); dut.clock.step() }
        axi.bready.poke(true.B); axi.bresp.expect(0.U)
        dut.clock.step()        // B handshake
        axi.bready.poke(false.B)
    }

    def read(addr: Int): BigInt = {
        for (_ <- 0 until rnd.nextInt(3)) dut.clock.step()  // random idle time
        axi.araddr.poke(addr.U); axi.arvalid.poke(true.B)
        var n = 0
        while(!axi.arready.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < maxWait, "no ARREADY") }
        dut.clock.step()        // AR handshake
        axi.arvalid.poke(false.B)
        n = 0
        while(!axi.rvalid.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < maxWait, "no RVALID")}
        for (_ <- 0 until rnd.nextInt(3)) { axi.rvalid.expect(true.B, "RVALID dropped before RREADY"); dut.clock.step() }
        val d = axi.rdata.peek().litValue
        axi.rresp.expect(0.U)
        axi.rready.poke(true.B); dut.clock.step(); axi.rready.poke(false.B)     // R handshake
        d
    }

    // send one packet on s_axis, collect m_axis with random backpressure (stall = true)
    // or at full speed (stall = false: TVALID whenever data is left, TREADY always high).
    // returns (received data, indexes of beats that had TLAST,
    //          cycle of the first s_axis handshake, cycle of the m_axis handshake with TLAST)
    def stream(in: Seq[BigInt], stall: Boolean = true): (Seq[BigInt], Seq[Int], Int, Int) = {
        val sIn = dut.io.s_axis; val mOut = dut.io.m_axis
        var sent = 0; var holding = false       // holding = TVALID high, wait for TREADY
        val got = scala.collection.mutable.ArrayBuffer[BigInt]()
        val lasts = scala.collection.mutable.ArrayBuffer[Int]()
        var cyc = 0
        var firstIn = -1; var lastOut = -1
        while(got.length < in.length) {
            if(!holding && sent < in.length && (!stall || rnd.nextInt(4) != 0)) holding = true
            sIn.tvalid.poke(holding.B)
            if(sent < in.length) { sIn.tdata.poke(in(sent).U); sIn.tlast.poke((sent == in.length - 1).B) }
            val ready = !stall || rnd.nextInt(3) != 0     // ready is true 2/3 of the cycles when stalling
            mOut.tready.poke(ready.B)
            val inFire = holding && sIn.tready.peek().litToBoolean
            val outFire = ready && mOut.tvalid.peek().litToBoolean
            if(inFire && firstIn < 0) firstIn = cyc
            if(outFire) {
                if(mOut.tlast.peek().litToBoolean) { lasts += got.length; lastOut = cyc }
                got += mOut.tdata.peek().litValue
            }
            dut.clock.step()
            if(inFire) { sent += 1; holding = false }
            cyc += 1
            assert(cyc < 20 * in.length + 100, "stream timeout")
        }
        sIn.tvalid.poke(false.B); mOut.tready.poke(false.B)
        (got.toSeq, lasts.toSeq, firstIn, lastOut)
    }
}

class AccelShellTest extends AnyFlatSpec with ChiselScalatestTester {
    behavior of "AccelShell"

    val cfg    = AccelConfig(core = "stub", nIn = 16, nOut = 16, streamW = 32)
    val mask32 = (BigInt(1) << 32) - 1

    it should "expose ID, config and the adder through AXI-Lite" in {
        test(new AccelShellInner(cfg)) { dut =>
            val bus = new AxiDriver(dut, new Random(1)); bus.idle()
            assert(bus.read(RegMap.ID) == cfg.id)
            assert(bus.read(RegMap.CFG_HASH) == cfg.cfgHash)
            assert(bus.read(RegMap.N_IN) == cfg.nIn)
            assert(bus.read(RegMap.N_OUT) == cfg.nOut)
            assert(bus.read(0xFFC) == 0, "unmapped address must return 0")

            val rnd = new Random(2)
            for(_ <- 0 until 50) {
                val a = rnd.nextLong() & 0xFFFFFFFFL; val b = rnd.nextLong() & 0xFFFFFFFFL
                bus.write(RegMap.ARG0, a); bus.write(RegMap.ARG0 + 4, b)
                assert(bus.read(RegMap.ARG0) == a)
                assert(bus.read(RegMap.ARG0 + 4) == b)
                assert(bus.read(RegMap.SUM) == ((BigInt(a) + BigInt(b)) & mask32))
            }
        }
    }

    it should "respect WSTRB byte enables" in {
        test(new AccelShellInner(cfg)) { dut =>
            val bus = new AxiDriver(dut, new Random(3)); bus.idle()
            bus.write(RegMap.ARG0 + 8, 0x11223344L)
            bus.write(RegMap.ARG0 + 8, 0xAABBCCDDL, strb = 0x5) // only bytes 0 and 2
            assert(bus.read(RegMap.ARG0 + 8) == BigInt("11BB33DD", 16))
        }
    }

    it should "stream data through the stub core with TLAST and backpressure" in {
        test(new AccelShellInner(cfg)) { dut =>
            val bus = new AxiDriver(dut, new Random(4)); bus.idle()
            val addend = 0x1000L
            bus.write(RegMap.ARG0, addend)
            bus.write(RegMap.CTRL, 1)
            assert((bus.read(RegMap.STATUS) & 2) == 0, "done must clear on start")

            val rnd = new Random(5)
            for (len <- Seq(1, 7, 64)) {
                bus.write(RegMap.CTRL, 1)
                val in = Seq.fill(len)(BigInt(32, rnd))     // random 32-bit values
                val (out, lasts, _, _) = bus.stream(in)
                assert(out == in.map(x => (x + addend) & mask32), s"data mismatch for len=$len")
                assert(lasts == Seq(len - 1), s"TLAST must be on the last beat only, got $lasts")
                assert((bus.read(RegMap.STATUS) & 2) == 2, "done must be set after the packet")
                assert(bus.read(RegMap.RESULT) == len)
                assert(bus.read(RegMap.CYCLES) > 0)
            }
        }
    }

    it should "count CYCLES from the first input beat to the last output beat" in {
        test(new AccelShellInner(cfg)) { dut =>
            val bus = new AxiDriver(dut, new Random(6)); bus.idle()
            val rnd = new Random(7)
            // Full speed: n beats through the stub's 1-cycle FIFO take exactly n cycles
            for (len <- Seq(1, 2, 7, 64)) {
                bus.write(RegMap.CTRL, 1)
                assert(bus.read(RegMap.CYCLES) == 0, "start must clear CYCLES")
                val (_, _, first, last) = bus.stream(Seq.fill(len)(BigInt(32, rnd)), stall = false)
                assert(last - first == len, s"testbench sanity: expected $len cycles, measured ${last - first}")
                assert(bus.read(RegMap.CYCLES) == len, s"CYCLES for len=$len at full speed")
            }
            // Random backpressure: CYCLES must match the testbench's own measurement exactly
            for (len <- Seq(1, 7, 64)) {
                bus.write(RegMap.CTRL, 1)
                val (_, _, first, last) = bus.stream(Seq.fill(len)(BigInt(32, rnd)))
                assert(bus.read(RegMap.CYCLES) == last - first, s"CYCLES for len=$len with stalls")
            }
            // Without start: the next packet is still measured (counting starts at its first beat)
            val (_, _, first, last) = bus.stream(Seq.fill(5)(BigInt(32, rnd)), stall = false)
            assert(bus.read(RegMap.CYCLES) == last - first)
        }
    }

    it should "keep m_axis tdata stable while stalled, even if ARG0 changes" in {
        test(new AccelShellInner(cfg)) { dut =>
            val bus = new AxiDriver(dut, new Random(8)); bus.idle()
            bus.write(RegMap.ARG0, 0x10L)
            // One beat in, then nothing; the receiver is not ready (tready low from idle())
            dut.io.s_axis.tdata.poke(5.U); dut.io.s_axis.tlast.poke(true.B); dut.io.s_axis.tvalid.poke(true.B)
            while (!dut.io.s_axis.tready.peek().litToBoolean) dut.clock.step()
            dut.clock.step()
            dut.io.s_axis.tvalid.poke(false.B)
            var n = 0
            while (!dut.io.m_axis.tvalid.peek().litToBoolean) { dut.clock.step(); n += 1; assert(n < 10) }
            dut.io.m_axis.tdata.expect(0x15.U)
            // CPU changes ARG0 while the beat is waiting: TVALID and TDATA must not change
            bus.write(RegMap.ARG0, 0x9999L)
            dut.io.m_axis.tvalid.expect(true.B, "TVALID dropped while stalled")
            dut.io.m_axis.tdata.expect(0x15.U, "TDATA changed while stalled")
            // Accept it; the next beat uses the new ARG0
            dut.io.m_axis.tready.poke(true.B); dut.clock.step(); dut.io.m_axis.tready.poke(false.B)
            val (out, _, _, _) = bus.stream(Seq(BigInt(1)))
            assert(out == Seq(BigInt(0x999A)))
        }
    }

    it should "resolve the stub config so the manifest has nOut = nIn" in {
        val r = AccelConfig.resolve(AccelConfig(core = "stub", nIn = 784, nOut = 10))
        assert(r.nOut == 784 && r.nIn == 784)
        // mlp has its own rule (sizes + weight hash from the network), tested in MlpCoreTest
        val other = AccelConfig(core = "cnn", nIn = 784, nOut = 10)
        assert(AccelConfig.resolve(other) == other, "cores without a rule are left unchanged")
    }

    it should "emit Verilog with the port names Vivado infers interfaces from" in {
        val full = circt.stage.ChiselStage.emitSystemVerilog(new AccelShell(cfg))
        val start = full.indexOf("module AccelShell(")      // look only at the top module's ports
        assert(start >= 0, "top module AccelShell not found")
        val sv = full.substring(start, full.indexOf(");", start))
        for (p <- Seq("aclk", "aresetn", "s_axi_awaddr", "s_axi_wstrb", "s_axi_bresp", "s_axi_rdata",
                      "s_axis_tdata", "s_axis_tready", "s_axis_tlast", "m_axis_tdata", "m_axis_tlast")) {
            assert(sv.contains(p), s"missing port $p")
        }
        assert(!sv.contains("io_s_axi"), "top-level ports must not carry the io_ prefix")
        val wrapper = Wrapper.verilog(cfg)
        assert(wrapper.contains("module accel_top") && wrapper.contains("AccelShell inner"))
        // every wrapper port must exist exactly once (catches copy-paste mistakes in Wrapper.ports)
        val names = Wrapper.ports(cfg).map(_._1)
        assert(names.distinct.length == names.length, s"duplicate wrapper ports: ${names.diff(names.distinct)}")
        for (n <- names) assert(sv.contains(n), s"wrapper port $n does not exist on AccelShell")
        assert(wrapper.contains(");\nendmodule"), "instance in wrapper must end with );")
    }
}