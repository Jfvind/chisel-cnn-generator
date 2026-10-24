# FPGA flow (PYNQ-Z2)

```
Chisel (shell.ShellGen) ──► hw/vivado/rtl/       AccelShell.sv, accel_top.v, shell_params.tcl, hw_manifest.json
Vivado (build.tcl)      ──► hw/vivado/overlay/   accel.bit, accel.hwh, hw_manifest.json   (+ reports/)
deploy.py               ──► pynq:~/jupyter_notebooks/cnn_accel/  and runs pynq/smoke_test.py
```

`hw/vivado/` is git-ignored. Only the core inside the shell changes between the adder test and the MLP;
the block design, register map, driver and scripts stay the same.

## System view on the board

![PS/PL system view and data flow of one Accel.run()](../docs/figures/pynq_system_dataflow.svg)

*Source: [`docs/figures/pynq_system_dataflow.tex`](../docs/figures/pynq_system_dataflow.tex). The inside of
the `accel` block is drawn in [`shell/README.md`](../generator/src/main/scala/shell/README.md).*

The PS (ARM + DDR, hard silicon) and the PL (our bitstream) talk over two kinds of AXI port:

| Port | Master | Used for | In this design |
|---|---|---|---|
| **M_AXI_GP0** (general purpose) | PS (the ARM) | Few, small register reads/writes (AXI-Lite) | ARM → `accel` registers, ARM → DMA control registers |
| **S_AXI_HP0** (high performance) | PL (the DMA) | Long bursts straight to the DDR controller, bypassing the CPU caches | DMA reads `ibuf` and writes `obuf` |

So the CPU only *controls*: it writes the input into DDR, starts the core and programs the DMA
(address + length, steps 1-3). The DMA moves the data itself over HP0 and AXI-Stream (steps 4-5), and
the CPU picks up the result when the DMA reports idle (step 6). Because HP0 bypasses the caches,
`accel.py` calls `flush()` before and `invalidate()` after each transfer.

PYNQ knows how to drive the DMA because `accel.hwh` lists it as `xilinx.com:ip:axi_dma` with its base
address, and `pynq.lib.dma.DMA` is a ready-made driver for exactly that IP. Our `accel` block has no
built-in driver, so `accel.py` maps its registers with `MMIO` and the offsets from `hw_manifest.json`.

## Prerequisites (checked 2026-10-14)

| What | Status |
|---|---|
| Vivado | 2024.1 installed in `C:\Xilinx\Vivado\2024.1`. Matches PYNQ v3.1 images (built with 2024.1). If the board runs v3.0.x (built with 2022.1), simple overlays like this one normally still load; if `Overlay()` fails parsing the `.hwh`, that mismatch is the first suspect. `smoke_test.py` prints the board's PYNQ version. |
| PYNQ-Z2 board files | Were **not** installed in Vivado. Now vendored in `hw/boards/TUL/pynq-z2` (from Xilinx/XilinxBoardStore) and loaded by `build.tcl` via `board.repoPaths`, so nothing needs installing. |
| Board on the network | Router: `pynq` via DHCP/DNS, Jupyter at `http://pynq:9090`. Direct cable: the board always also has `192.168.2.99` (alias `eth0:1`), Jupyter at `http://192.168.2.99:9090`. `deploy.py` tries `pynq-direct`, then `pynq`. |
| SSH key + sudo | One-time, see below. |

### One-time board setup (from PowerShell)

```powershell
ssh-keygen                                    # skip if you already have ~/.ssh/id_ed25519
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh xilinx@pynq "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
# Optional, lets build.tcl run the test without a password prompt:
ssh -t xilinx@pynq "echo 'xilinx ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/xilinx-nopasswd"
```

### Direct cable to the PC (no router)

Give the PC's Ethernet adapter a fixed address in the board's subnet, without a gateway
(internet stays on Wi-Fi). Admin PowerShell, adapter name from `Get-NetAdapter`:

```powershell
Set-NetIPInterface -InterfaceAlias "Ethernet 5" -Dhcp Disabled
New-NetIPAddress   -InterfaceAlias "Ethernet 5" -IPAddress 192.168.2.1 -PrefixLength 24
```

Add `pynq-direct` (HostName 192.168.2.99, User xilinx) to `~/.ssh/config`. On the cable
the board has no internet, so install packages on the board while it is on the router.
Back to DHCP: `Set-NetIPInterface -InterfaceAlias "Ethernet 5" -Dhcp Enabled`.

## Running it

With `make` (Linux, or Windows with GNU make). The Makefile picks `vivado.bat`/`sbt.bat`/`python` on Windows
and `vivado`/`sbt`/`python3` on Linux; Vivado's `bin` folder must be on PATH. `make help` shows what it will
use; override with e.g. `make bitstream VIVADO=C:/Xilinx/Vivado/2024.1/bin/vivado.bat`.

```bash
make test        # chiseltest, incl. AXI-Lite/AXIS bus-model tests of the shell
make board       # rtl -> bitstream -> deploy -> smoke test on the board
```

Without `make` (PowerShell, from the repo root):

```powershell
cd generator; sbt "runMain shell.ShellGen --out ../hw/vivado/rtl --core stub --streamW 32"; cd ..
C:\Xilinx\Vivado\2024.1\bin\vivado.bat -mode batch -nojournal -log hw/vivado/vivado.log `
    -source hw/scripts/build.tcl -tclargs hw/vivado/rtl hw/vivado accel 8 1
```

The last argument `1` makes `build.tcl` call `deploy.py` at the end, which copies the overlay and the
driver to the board and runs `smoke_test.py`. Use `0` to only build, then
`python hw/scripts/deploy.py --overlay hw/vivado/overlay/accel.bit`.

Expected board output ends with `ALL PASSED`. The overlay and `smoke_test.py` also land in
`cnn_accel/` in Jupyter, so you can poke registers interactively with `from accel import Accel`.

## Register map (AXI-Lite, identical for every core)

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x00 | CTRL | W | bit0 start (pulse), bit1 soft-reset core |
| 0x04 | STATUS | R | bit0 busy, bit1 done (sticky, cleared by start) |
| 0x08 | ID | R | 0xDEADBEEF (`AccelConfig.id`) |
| 0x0C | CFG_HASH | R | must equal `cfg_hash` in hw_manifest.json |
| 0x10 / 0x14 | N_IN / N_OUT | R | packet sizes |
| 0x18 | CYCLES | R | clock cycles from the first `s_axis` beat to the `m_axis` beat with TLAST (start resets it) |
| 0x1C | OVF_COUNT | R | activations that saturated (core-defined) |
| 0x20 | RESULT | R | core-defined (MLP: argmax; stub: beats in last packet) |
| 0x40–0x4C | ARG0–ARG3 | RW | passed to the core as `io.args` |
| 0x50 | SUM | R | ARG0 + ARG1 (adder smoke test) |

## The MLP core

`MlpCore` (`generator/src/main/scala/mlp/`) runs the whole network on one column of MACs, one per
neuron of the widest layer; layer 0 consumes the pixels straight from the stream. 925 cycles per image
for 784 -> 128 -> 10 at full speed.

```bash
make board CORE=mlp                         # 32-bit beats: 784 pixels in, 10 signed scores out
make board CORE=mlp SHIFT_POLICY=typical    # more activation precision, some activations saturate
```

The weights are quantized from `WEIGHTS` (default `python/models/csv`) with `WEIGHT_BITS` /
`ACTIVATION_BITS` / `SHIFT_POLICY` and baked into BRAM (one bank per lane, `$readmemh`). They are part of
`CFG_HASH`, so a manifest from other weights is rejected by the driver. `RESULT` = argmax,
`OVF_COUNT` = hidden activations that saturated since the last start (always 0 with `SHIFT_POLICY=worst`).
