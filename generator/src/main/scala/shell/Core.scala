package shell

import chisel3._
import chisel3.util._

// =============================================================================
// Core.scala — the contract between the shell and any accelerator core.
//
// CoreIO     : in/out streams as Decoupled(StreamBeat), start pulse, ARG0..3,
//              busy, done pulse, result, ovf. The MLP core implements this.
// AccelCore  : abstract base class (Module with io = CoreIO) + factory that picks
//              a core by name ("stub" now, "mlp" later).
// StreamStub : smoke-test core: out = in + ARG0 per beat (added before the FIFO), TLAST passed through,
//              done pulses on the last output beat, RESULT = beats in last packet.
// =============================================================================

class CoreIO(cfg: AccelConfig) extends Bundle {
    val in     = Flipped(Decoupled(new StreamBeat(cfg.streamW)))  // core is the sink, flipped
    val out    = Decoupled(new StreamBeat(cfg.streamW))           // core is the Source
    val start  = Input(Bool())                                    // from CTRL bit0
    val args   = Input(Vec(RegMap.NUM_ARGS, UInt(32.W)))          // ARG0-ARG3
    val busy   = Output(Bool())
    val done   = Output(Bool())                                   // Inference finished
    val result = Output(UInt(32.W))                               // RESULT register
    val ovf    = Output(UInt(32.W))                               // OVF_COUNT register
}

// Base class. All core modules must have IOs matching CoreIO
abstract class AccelCore(val cfg: AccelConfig) extends Module {
    val io = IO(new CoreIO(cfg))
}

// Core selector. Passed as input argument
object AccelCore {
    def apply(cfg: AccelConfig): AccelCore = cfg.core match {
        case "stub" => new StreamStub(cfg)
        case "mlp"  => new mlp.MlpCore(cfg, service.WeightLoader.load(cfg.weightBiasDir))
        case other => throw new IllegalArgumentException(s"Unknown core '$other'")
    }
}

class StreamStub(cfg: AccelConfig) extends AccelCore(cfg) {
    // Add BEFORE the FIFO: the FIFO then holds the finished value, so m_axis tdata comes
    // straight from a register and stays stable while the receiver stalls (AXI-Stream rule),
    // even if the CPU writes a new ARG0 in the meantime. ARG0 is sampled when a beat enters.
    val added = Wire(Decoupled(new StreamBeat(cfg.streamW)))
    added.valid     := io.in.valid
    added.bits.data := io.in.bits.data + io.args(0)    // Adder logic for the test
    added.bits.last := io.in.bits.last                 // forward TLAST
    io.in.ready     := added.ready

    val q = Queue(added, 2)                             // 2-entry FIFO

    io.out.valid := q.valid                             // Output valid when FIFO has a beat
    io.out.bits  := q.bits                              // registered data + last, no logic after the FIFO
    q.ready := io.out.ready                             // pop the FIFO when downstream is ready to receive

    val beats     = RegInit(0.U(32.W))                  // Beats in current packet
    val lastCount = RegInit(0.U(32.W))                  // Length of last finished packet
    val active    = RegInit(false.B)                    // When working (between start and end)

    val outFire = io.out.valid && io.out.ready          // Beat leaves in this cycle
    when(io.start) { active := true.B }
    when(outFire) {
        beats := Mux(io.out.bits.last, 0.U, beats + 1.U) // count restarts after last beat
        when(io.out.bits.last) {
            lastCount := beats + 1.U   // +1 includes the last beat
            active := false.B
        }
    }

    io.busy := active || q.valid
    io.done := outFire && io.out.bits.last          // Pulse when last beat leaves
    io.result := lastCount
    io.ovf    := 0.U                               // Stub doesn't saturate
}