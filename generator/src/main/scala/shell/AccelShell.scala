package shell

import chisel3._
import chisel3.util._

// =============================================================================
// AccelShell.scala — the shell around the core.
//
// AccelShellInner : normal Module (implicit clock, active-high reset) so it can be
//                   tested with chiseltest. Contains the AXI-Lite slave, the register
//                   file (RegMap), the cycle counter, the AXIS <-> Decoupled wiring and
//                   the core (with soft reset).
// AccelShell      : RawModule top with the names Vivado expects: aclk, aresetn
//                   (active low), s_axi_*, s_axis_*, m_axis_*.
// =============================================================================

// All except clock and reset.
class AccelShellInner(cfg: AccelConfig) extends Module {
    val io = IO(new Bundle {
        val s_axi = new AxiLiteSlaveIO(cfg.addrW)       // registers
        val s_axis = Flipped(new AxisIO(cfg.streamW))   // data in (we are stream slave)
        val m_axis = new AxisIO(cfg.streamW)            // data out (we are master)
    })

    // --- AXI-Lite protocol engine ---
    val axil = Module(new AxiLiteSlave(cfg.addrW))
    axil.io.axi <> io.s_axi
    val wr = axil.io.wr         // shorter more convenient write name

    def wordAddr(byteAddr: Int): UInt = (byteAddr >> 2).U
    val wrWord = wr.bits.addr >> 2                      // byte-select bits removed, focus on word only
    val rdWord = axil.io.rdAddr >> 2
    val wrMask = FillInterleaved(8, wr.bits.strb)       // strb b0101 -> 0x00FF00FF. Fills a byte with 1s if strobe bit is high for that byte

    // --- ARG0-ARG3 r/w registers ---
    val args = RegInit(VecInit(Seq.fill(RegMap.NUM_ARGS)(0.U(32.W))))   // registers initialized to 0.U
    for (i <- 0 until RegMap.NUM_ARGS) {
        when(wr.valid && wrWord === wordAddr(RegMap.ARG0 + 4 * i)) {
            args(i) := (args(i) & ~wrMask) | (wr.bits.data & wrMask)    // writing only the enabled bytes
        }
    }

    // --- CTRL w ---
    val ctrlWrite = wr.valid && wrWord === wordAddr(RegMap.CTRL) && wr.bits.strb(0) // writing to the lowest byte
    val start     = ctrlWrite && wr.bits.data(0)                            // bit0, start
    val softReset = RegNext(ctrlWrite && wr.bits.data(1), false.B)          // bit1, reset

    // --- Core ---
    val core = withReset(reset.asBool || softReset) { Module(AccelCore(cfg)) }
    core.io.start := start
    core.io.args  := args

    // --- AXI-Stream ---
    core.io.in.valid     := io.s_axis.tvalid
    core.io.in.bits.data := io.s_axis.tdata
    core.io.in.bits.last := io.s_axis.tlast
    io.s_axis.tready     := core.io.in.ready

    io.m_axis.tvalid     := core.io.out.valid
    io.m_axis.tdata      := core.io.out.bits.data
    io.m_axis.tlast      := core.io.out.bits.last
    core.io.out.ready    := io.m_axis.tready

    // --- STATUS.done: sticky, set by the core's done pulse, cleared by start
    val done = RegInit(false.B)
    when(start) {
        done := false.B
    }.elsewhen(core.io.done) {
        done := true.B
    }

    // --- CYCLES: pure accelerator time, independent of Python and DMA setup.
    // Definition: CYCLES = t_last - t_first, where t_first is the cycle of the first s_axis
    // handshake of a packet and t_last the cycle of the m_axis handshake carrying TLAST.
    // (n beats through a 1-cycle pipeline with no stalls gives exactly n.)
    val inFire   = core.io.in.valid && core.io.in.ready                               // s_axis beat accepted
    val outLast  = core.io.out.valid && core.io.out.ready && core.io.out.bits.last    // last m_axis beat leaves
    val counting = RegInit(false.B)
    val cycles   = RegInit(0.U(32.W))

    when(start) { cycles := 0.U; counting := false.B }           // start only resets; counting begins with the data
    when(counting) { cycles := cycles + 1.U }                     // includes the cycle of the last beat
    when(counting && outLast) { counting := false.B }             // freeze until the next packet
    when(!counting && inFire) {                                   // first input beat of a packet
        cycles   := 0.U
        counting := !outLast                                      // (in and last out in the same cycle: 0)
    }

    // --- Read mux: word index -> value
    val readMap: Seq[(UInt, UInt)] = Seq(
        wordAddr(RegMap.STATUS) -> Cat(done, core.io.busy),      // bit1 = done, bit0 = busy
        wordAddr(RegMap.ID)     -> cfg.id.U(32.W),
        wordAddr(RegMap.CFG_HASH) -> cfg.cfgHash.U(32.W),
        wordAddr(RegMap.N_IN)     -> cfg.nIn.U(32.W),
        wordAddr(RegMap.N_OUT)    -> cfg.nOut.U(32.W),
        wordAddr(RegMap.CYCLES)   -> cycles,
        wordAddr(RegMap.OVF_COUNT) -> core.io.ovf,
        wordAddr(RegMap.RESULT)    -> core.io.result,
        wordAddr(RegMap.SUM)       -> (args(0) + args(1))       // adder
    ) ++ (0 until RegMap.NUM_ARGS).map(i => wordAddr(RegMap.ARG0 + 4 * i) -> args(i)) // Args read-back

    // Unmapped address return 0, entries are padded to 32-bit
    axil.io.rdData := MuxLookup(rdWord, 0.U(32.W))(readMap.map { case (a, d) => (a, d.pad(32)) })
}

// Top-level for Vivado
class AccelShell(cfg: AccelConfig) extends RawModule {      // RawModule bears no implicit clock or reset
    val aclk = IO(Input(Clock()))
    val aresetn = IO(Input(Bool()))
    val s_axi   = IO(new AxiLiteSlaveIO(cfg.addrW))
    val s_axis  = IO(Flipped(new AxisIO(cfg.streamW)))
    val m_axis  = IO(new AxisIO(cfg.streamW))

    // All inside runs aclk and active low reset
    val inner = withClockAndReset(aclk, !aresetn) { Module(new AccelShellInner(cfg)) }
    inner.io.s_axi  <> s_axi
    inner.io.s_axis <> s_axis
    inner.io.m_axis <> m_axis
}