"""accel.py — generic PYNQ driver for any core inside AccelShell.

Everything design-specific (register offsets, sizes, stream width, id/hash) comes from
hw_manifest.json, written by the Chisel generator next to the bitstream. The same driver
works for the stream stub and for the MLP.

    from accel import Accel
    acc = Accel("accel.bit")
    out, cycles = acc.run(x)          # x: numpy array, one packet

Accel: loads the overlay, checks ID/CFG_HASH, register read/write helpers,
       run() = one packet through the AXI DMA.
"""

import json
import pathlib

import numpy as np
from pynq import MMIO, Overlay, allocate

_DTYPES = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}    # stream width becomes numpy type

class Accel:
    def __init__(self, bitfile="accel.bit", ip_name="accel", dma_name="axi_dma_0", download=True):
        bitfile = pathlib.Path(bitfile).resolve()
        self.manifest = json.loads((bitfile.parent / "hw_manifest.json").read_text())
        self.r = self.manifest["regs"]      # {"CTRL": 0, ...}
        self.dtype = _DTYPES[self.manifest["stream_w"]]

        self.ol = Overlay(str(bitfile), download=download)  # programs PL, parses .hwh
        ip = self.ol.ip_dict[ip_name]       # "accel" from cell name in block design
        self.mmio = MMIO(ip["phys_addr"], ip["addr_range"])     # maps the 4KB register space
        self.dma = getattr(self.ol, dma_name)       # pynq adds its DMA driver automatically

        hw_id, hw_hash = self.read("ID"), self.read("CFG_HASH")
        if hw_id != self.manifest["id"]:
            raise RuntimeError(f"ID register 0x{hw_id:08x} != manifest 0x{self.manifest['id']:08x}: wrong bitstream?")
        if hw_hash != self.manifest["cfg_hash"]:
            raise RuntimeError("CFG_HASH mismatch: bitstream and hw_manifest.json come from different builds")
        self._bufs = {}     # reusing DMA buffers between runs

    # registers accept either a name ("SUM") or an offset (0x50)
    def read(self, reg):
        return self.mmio.read(self.r[reg] if isinstance(reg, str) else reg)

    def write(self, reg, value):
        self.mmio.write(self.r[reg] if isinstance(reg, str) else reg, int(value) & 0xFFFFFFFF)

    def set_arg(self, i, value):
        self.write(self.r["ARG0"] + 4 * i, value)

    def start(self):
        self.write("CTRL", 1)

    def soft_reset(self):
        self.write("CTRL", 2)

    @property
    def status(self):
        s = self.read("STATUS")
        return {"busy": bool(s & 1), "done": bool(s & 2)}

    # DMA buffers must be physically contiguous: pynq.allocate
    def _buf(self, key, n):
        b = self._bufs.get(key)
        if b is None or b.shape[0] != n:
            if b is not None:
                b.freebuffer()
            b = allocate(shape=(n,), dtype=self.dtype)
            self._bufs[key] = b
        return b

    def run(self, x, n_out=None):
        """One packet in, one packet out. Without n_out the packet must be exactly n_in long
        and n_out comes from the manifest; pass n_out for any other packet length."""
        x = np.asarray(x)
        if n_out is None:
            if x.size != self.manifest["n_in"]:
                raise ValueError(f"packet has {x.size} values but the core expects n_in={self.manifest['n_in']}; "
                                 f"pass n_out explicitly for other packet lengths")
            n_out = self.manifest["n_out"]
        ibuf, obuf = self._buf("in", x.size), self._buf("out", n_out)
        ibuf[:] = x.astype(self.dtype, copy=False).ravel()
        ibuf.flush()                                # writes CPU cache to DDR so it's visible to the DMA

        self.start()
        self.dma.recvchannel.transfer(obuf)         # ready S2MM first so no output beat is lost
        self.dma.sendchannel.transfer(ibuf)         # MM2S: DDR -> s_axis
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()                 # returns when core sends TLAST
        obuf.invalidate()                           # Drops stale cachelines so the CPU sees the new ones.
        return np.array(obuf), self.read("CYCLES")   # CYCLES: first input beat -> last output beat

    def close(self):
        for b in self._bufs.values():
            b.freebuffer()
        self._bufs.clear()