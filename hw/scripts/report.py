#!/usr/bin/env python3
"""report.py — turn Vivado build reports into one readable HTML dashboard.

    python hw/scripts/report.py                         # hw/vivado -> hw/vivado/reports/dashboard.html
    python hw/scripts/report.py hw/vivado/a hw/vivado/b # compare builds -> hw/vivado/compare.html
    python hw/scripts/report.py --open                  # also open it in the browser

A build directory is what build.tcl writes (out_dir):
    reports/utilization.rpt, reports/utilization_hier.rpt, reports/timing_summary.rpt
    rtl/shell_params.tcl, overlay/hw_manifest.json        (config, optional)
    reports/board_results.json                             (board metrics, optional; mlp_test.py)
    python/results/cpu_benchmark.json                       (PC CPU timing, optional; make cpu-bench)

Standard library only, so it runs on Windows and Linux without installing anything.
Missing files are skipped; the page shows what it could read.
"""

### OBS: AI-gen dashboard - data from vivado reports ###

import argparse
import datetime
import html
import json
import math
import pathlib
import re
import sys
import webbrowser

# ----------------------------------------------------------------------------- parsing

def read(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def num(s):
    """'2731' -> 2731, '2.5' -> 2.5, '' / '-' -> None."""
    s = s.strip().rstrip("*").strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def header_meta(text):
    """'| Date : ...' lines at the top of every Vivado report."""
    meta = {}
    for m in re.finditer(r"^\|\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*$", text or "", re.M):
        meta.setdefault(m.group(1), m.group(2))
    return meta


def tables(text):
    """Every +---+ bordered table as (headers, rows); cells keep their leading spaces."""
    out, lines, i = [], (text or "").splitlines(), 0
    while i < len(lines):
        if lines[i].startswith("+-") and i + 2 < len(lines) and lines[i + 1].startswith("|") \
                and lines[i + 2].startswith("+-"):
            headers = [c.strip() for c in lines[i + 1].strip().strip("|").split("|")]
            rows, i = [], i + 3
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i].rstrip()[1:-1].split("|"))
                i += 1
            out.append((headers, rows))
        i += 1
    return out


def utilization(text):
    """{'Slice LUTs': (used, available, util%), ...} — first occurrence of each site type."""
    res = {}
    for headers, rows in tables(text):
        if "Site Type" not in headers or "Used" not in headers or "Available" not in headers:
            continue
        iu, ia = headers.index("Used"), headers.index("Available")
        ip = next((k for k, h in enumerate(headers) if h.startswith("Util")), None)
        for r in rows:
            if len(r) != len(headers):
                continue
            name = r[0].strip().rstrip("*").strip()
            res.setdefault(name, (num(r[iu]), num(r[ia]), num(r[ip]) if ip is not None else None))
    return res


TIMING_FIELDS = ["WNS", "TNS", "TNS failing", "TNS total", "WHS", "THS", "THS failing", "THS total",
                 "WPWS", "TPWS", "TPWS failing", "TPWS total"]


def timing(text):
    """Design Timing Summary values + clocks from report_timing_summary."""
    t = {"clocks": []}
    if not text:
        return t
    start = text.find("Design Timing Summary")
    lines = text[start:].splitlines() if start >= 0 else []
    for k, line in enumerate(lines):
        if "WNS(ns)" in line and k + 2 < len(lines):
            vals = lines[k + 2].split()
            if len(vals) >= len(TIMING_FIELDS):
                t.update({f: num(v) for f, v in zip(TIMING_FIELDS, vals)})
            break
    t["met"] = "All user specified timing constraints are met" in text
    cs = text.find("Clock Summary")
    if cs >= 0:
        for m in re.finditer(r"^(\S+)\s+\{([^}]*)\}\s+([\d.]+)\s+([\d.]+)\s*$", text[cs:cs + 4000], re.M):
            t["clocks"].append({"name": m.group(1), "period": float(m.group(3)), "mhz": float(m.group(4))})
    return t


def hierarchy(text):
    """Rows of report_utilization -hierarchical as dicts with a 'depth' from the indentation."""
    for headers, rows in tables(text):
        if "Instance" not in headers or "Module" not in headers:
            continue
        out = []
        for r in rows:
            if len(r) != len(headers):
                continue
            raw = r[0]
            d = {"depth": (len(raw) - len(raw.lstrip(" ")) - 1) // 2,
                 "instance": raw.strip(), "module": r[1].strip()}
            for h, c in zip(headers[2:], r[2:]):
                d[h] = num(c) or 0
            out.append(d)
        return out
    return []


def tcl_params(text):
    return {m.group(1): m.group(2) for m in re.finditer(r'^set\s+(\w+)\s+"?([^"\n]*?)"?\s*$', text or "", re.M)}


def load_build(d):
    d = pathlib.Path(d)
    rep = d / "reports"
    util_txt, tim_txt, hier_txt = (read(rep / f) for f in
                                   ("utilization.rpt", "timing_summary.rpt", "utilization_hier.rpt"))
    manifest = read(d / "overlay" / "hw_manifest.json") or read(d / "rtl" / "hw_manifest.json")
    board = read(rep / "board_results.json")
    b = {
        "dir": d, "name": d.name or str(d),
        "util": utilization(util_txt), "timing": timing(tim_txt), "hier": hierarchy(hier_txt),
        "meta": header_meta(util_txt or tim_txt),
        "params": tcl_params(read(d / "rtl" / "shell_params.tcl")),
        "manifest": json.loads(manifest) if manifest else {},
        "board": json.loads(board) if board else None,
        "found": [f for f in ("utilization.rpt", "timing_summary.rpt", "utilization_hier.rpt") if (rep / f).exists()],
    }
    b["summary"] = summarize(b)
    return b


def summarize(b):
    """The handful of numbers every view needs."""
    u, t, p, m = b["util"], b["timing"], b["params"], b["manifest"]
    s = {"core": m.get("core") or p.get("CORE") or "?", "stream_w": m.get("stream_w") or p.get("STREAM_W"),
         "n_in": m.get("n_in"), "n_out": m.get("n_out")}
    fclk = m.get("fclk_mhz") or p.get("FCLK_MHZ")
    period = t["clocks"][0]["period"] if t["clocks"] else (1000.0 / float(fclk) if fclk else None)
    s["fclk"] = float(fclk) if fclk else (t["clocks"][0]["mhz"] if t["clocks"] else None)
    s["period"], s["wns"], s["whs"] = period, t.get("WNS"), t.get("WHS")
    s["failing"] = (t.get("TNS failing") or 0) + (t.get("THS failing") or 0)
    s["met"] = t.get("met") if t.get("WNS") is not None else None
    s["fmax"] = 1000.0 / (period - s["wns"]) if period and s["wns"] is not None and period - s["wns"] > 0 else None
    for key, site in (("lut", "Slice LUTs"), ("ff", "Slice Registers"), ("bram", "Block RAM Tile"),
                      ("dsp", "DSPs"), ("lutram", "LUT as Memory")):
        s[key] = u.get(site)
    s["accel"] = accel_node(b["hier"])
    return s


def components(hier):
    """Children of the block design instance (depth 2), or depth 1 if the tree is shallow."""
    depth = 2 if any(r["depth"] == 2 for r in hier) else 1
    return [r for r in hier if r["depth"] == depth]


def accel_node(hier):
    """Our design: the instance wrapping accel_top / AccelShell."""
    for k, r in enumerate(hier):
        if r["module"] in ("accel_top", "AccelShell"):
            # walk up to the block-design cell that owns it (shallowest ancestor at component depth)
            comps = components(hier)
            owner = None
            for c in comps:
                if hier.index(c) <= k:
                    owner = c
            return owner or r
    return None

# ----------------------------------------------------------------------------- rendering

def esc(x):
    return html.escape(str(x))


def fmt(v, digits=None, unit=""):
    """Number with thousands separators; digits=None -> 0 for whole numbers, else 1."""
    if v is None:
        return "–"
    if digits is None:
        digits = 0 if isinstance(v, int) or (isinstance(v, float) and v.is_integer()) else 1
    s = f"{v:,.{digits}f}" if isinstance(v, (int, float)) else str(v)
    return s + unit


def level(pct):
    if pct is None:
        return "muted"
    return "bad" if pct >= 90 else "warn" if pct >= 70 else "ok"


CSS = """
:root{--bg:#f6f7f9;--panel:#fff;--ink:#16181d;--muted:#5f6673;--line:#e3e6eb;--track:#eceef2;
--ok:#1f8a5b;--ok-bg:#e3f4ec;--warn:#b26a00;--warn-bg:#fdf0db;--bad:#c2372f;--bad-bg:#fbe5e3;
--accent:#3156c9;--accent-bg:#e7ecfb;--other:#9aa3b2;font-synthesis:none}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--panel:#1a1d24;--ink:#e8eaf0;--muted:#9aa1ad;
--line:#2a2e37;--track:#262a33;--ok:#4cc38a;--ok-bg:#16301f;--warn:#e7a43c;--warn-bg:#33270f;
--bad:#f07167;--bad-bg:#3a1a18;--accent:#7f9cff;--accent-bg:#1e2747;--other:#5d6574}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0 0 12px}
.sub{color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 22px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:3px 10px;font-size:12.5px}
.chip b{font-weight:600}
.grid{display:grid;gap:12px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin-bottom:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpi .label{color:var(--muted);font-size:12.5px}
.kpi .value{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}
.kpi .hint{color:var(--muted);font-size:12px;margin-top:4px}
.pill{display:inline-block;white-space:nowrap;border-radius:999px;padding:2px 10px;font-size:12.5px;font-weight:600}
.pill.ok{background:var(--ok-bg);color:var(--ok)}.pill.bad{background:var(--bad-bg);color:var(--bad)}
.pill.warn{background:var(--warn-bg);color:var(--warn)}.pill.muted{background:var(--track);color:var(--muted)}
.two{grid-template-columns:1fr 1fr;margin-bottom:12px}
@media (max-width:820px){.two{grid-template-columns:1fr}}
.grid>*{min-width:0}
.bar{display:grid;grid-template-columns:112px 1fr 158px;align-items:center;gap:10px;margin:10px 0}
.bar .name{font-size:13px}.bar .nums{text-align:right;font-variant-numeric:tabular-nums;font-size:12.5px;color:var(--muted);white-space:nowrap}
.wide .bar{grid-template-columns:190px 1fr 90px}
@media (max-width:520px){.wide .bar,.bar{grid-template-columns:1fr auto;gap:4px 10px}.bar .track{grid-column:1/-1;grid-row:2}}
.track{height:10px;background:var(--track);border-radius:6px;overflow:hidden}
.fill{height:100%;border-radius:6px}.fill.ok{background:var(--ok)}.fill.warn{background:var(--warn)}
.fill.bad{background:var(--bad)}.fill.muted{background:var(--other)}
.stack{display:flex;height:22px;border-radius:6px;overflow:hidden;background:var(--track);margin:4px 0 12px}
.stack span{display:block;height:100%;border-right:2px solid var(--panel)}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:12.5px;color:var(--muted)}
.dot{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:12px}
th:first-child,td:first-child{text-align:left}
tr.hl td{background:var(--accent-bg)}
.scroll{overflow-x:auto}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted);font-size:13px}
.slack{position:relative;height:12px;background:var(--track);border-radius:6px;margin-top:10px;overflow:hidden}
.slack i{position:absolute;top:0;bottom:0;left:0;border-radius:6px}
.note{color:var(--muted);font-size:12.5px;margin-top:8px}
.run{margin-top:28px}
footer{margin-top:28px;color:var(--muted);font-size:12px}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:12px}
"""

PALETTE = ["var(--accent)", "#8b5cf6", "#0ea5a4", "#d97706", "#db2777", "#64748b", "#16a34a", "#9aa3b2"]


def bar(name, triple):
    if not triple:
        return ""
    used, avail, pct = triple
    if pct is None and used is not None and avail:
        pct = 100.0 * used / avail
    w = min(pct or 0, 100)
    return (f'<div class="bar"><div class="name">{esc(name)}</div>'
            f'<div class="track"><div class="fill {level(pct)}" style="width:{w:.2f}%"></div></div>'
            f'<div class="nums">{fmt(used)} / {fmt(avail)} · {fmt(pct, 1)}%</div></div>')


def kpi(label, value, hint="", pill=None):
    p = f'<span class="pill {pill[0]}">{esc(pill[1])}</span>' if pill else ""
    return (f'<div class="card kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value">{value}</div><div class="hint">{p} {hint}</div></div>')


def timing_cards(s):
    met = s["met"]
    status = ("ok", "Timing met") if met else ("bad", "Timing failed") if met is False else ("muted", "no timing report")
    period, wns = s["period"], s["wns"]
    slack_bar = ""
    if period and wns is not None:
        used = max(0.0, min(period, period - wns))
        cls = "ok" if wns >= 0.5 else "warn" if wns >= 0 else "bad"
        slack_bar = (f'<div class="slack"><i class="fill {cls}" style="width:{100 * used / period:.1f}%"></i></div>'
                     f'<div class="note">Critical path uses {used:.2f} of {period:.2f} ns '
                     f'({100 * used / period:.0f}% of the clock period).</div>')
    return (
        '<div class="grid kpis">'
        + kpi("Worst setup slack (WNS)", fmt(wns, 3, " ns"), "", status)
        + kpi("Estimated Fmax", fmt(s["fmax"], 0, " MHz"),
              f"at {fmt(s['fclk'], 0)} MHz target" if s["fclk"] else "")
        + kpi("Worst hold slack (WHS)", fmt(s["whs"], 3, " ns"), "must be ≥ 0")
        + kpi("Failing endpoints", fmt(s["failing"]), "setup + hold",
              ("ok", "none") if s["failing"] == 0 and met is not None else None)
        + "</div>" + (f'<div class="card" style="margin-bottom:12px"><h2>Clock period budget</h2>{slack_bar}'
                      '<div class="note">Estimated Fmax = 1 / (period − WNS). Vivado only proves the target '
                      'clock; the estimate says how much headroom is left, not a guaranteed frequency.</div></div>'
                      if slack_bar else "")
    )


def resources_card(b):
    u = b["util"]
    rows = [("LUTs", u.get("Slice LUTs")), ("Flip-flops", u.get("Slice Registers")),
            ("LUT as memory", u.get("LUT as Memory")), ("Block RAM (36k)", u.get("Block RAM Tile")),
            ("DSP slices", u.get("DSPs")), ("Slices", u.get("Slice"))]
    body = "".join(bar(n, t) for n, t in rows if t)
    if not body:
        body = '<div class="note">No utilization.rpt found.</div>'
    ram = u.get("RAMB36/FIFO"), u.get("RAMB18")
    extra = ""
    if any(ram):
        extra = (f'<div class="note">Block RAM tiles = RAMB36 + RAMB18/2: '
                 f'{fmt(ram[0][0] if ram[0] else 0)} × RAMB36, {fmt(ram[1][0] if ram[1] else 0)} × RAMB18.</div>')
    return f'<div class="card"><h2>Device utilization (xc7z020)</h2>{body}{extra}</div>'


def breakdown_card(b):
    hier = b["hier"]
    if not hier:
        return '<div class="card"><h2>Where the LUTs go</h2><div class="note">No utilization_hier.rpt found.</div></div>'
    comps = sorted(components(hier), key=lambda r: -r.get("Total LUTs", 0))
    total = sum(r.get("Total LUTs", 0) for r in comps) or 1
    acc = b["summary"]["accel"]
    colors = {}
    k = 1
    for r in comps:
        if acc and r is acc:
            colors[id(r)] = PALETTE[0]
        else:
            colors[id(r)] = PALETTE[k % len(PALETTE)] if k < len(PALETTE) - 1 else PALETTE[-1]
            k += 1
    stack = "".join(f'<span title="{esc(r["instance"])}: {fmt(r.get("Total LUTs"))} LUTs" '
                    f'style="width:{100 * r.get("Total LUTs", 0) / total:.2f}%;background:{colors[id(r)]}"></span>'
                    for r in comps)
    legend = "".join(f'<span><i class="dot" style="background:{colors[id(r)]}"></i>{esc(r["instance"])}</span>'
                     for r in comps)
    cols = ["Total LUTs", "FFs", "RAMB36", "RAMB18", "DSP Blocks"]
    rows = "".join(
        f'<tr class="{"hl" if acc and r is acc else ""}"><td>{esc(r["instance"])}</td>'
        + "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in comps)
    note = ('<div class="note">Highlighted: our accelerator (AccelShell + core). The rest is the fixed '
            'platform: AXI DMA, interconnects, reset and the PS wrapper.</div>') if acc else ""
    return (f'<div class="card"><h2>Where the LUTs go</h2><div class="stack">{stack}</div>'
            f'<div class="legend">{legend}</div><div class="scroll" style="margin-top:12px"><table>'
            f'<tr><th>Block</th><th>LUTs</th><th>FFs</th><th>RAMB36</th><th>RAMB18</th><th>DSP</th></tr>'
            f'{rows}</table></div>{note}</div>')


def hierarchy_details(b):
    hier = b["hier"]
    if not hier:
        return ""
    cols = [c for c in ("Total LUTs", "Logic LUTs", "LUTRAMs", "SRLs", "FFs", "RAMB36", "RAMB18", "DSP Blocks")
            if c in hier[0]]
    head = "<tr><th>Instance</th><th>Module</th>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"
    rows = "".join(
        f'<tr><td style="padding-left:{8 + 16 * r["depth"]}px">{esc(r["instance"])}</td>'
        f'<td style="text-align:left">{esc(r["module"])}</td>'
        + "".join(f"<td>{fmt(r.get(c))}</td>" for c in cols) + "</tr>" for r in hier)
    return (f'<div class="card" style="margin-top:12px"><details><summary>Full hierarchy '
            f'({len(hier)} instances, from utilization_hier.rpt)</summary><div class="scroll" '
            f'style="margin-top:10px"><table>{head}{rows}</table></div></details></div>')


def fmt_time(ms):
    """Milliseconds -> readable with the right unit: 0.00925 -> '9.25 µs'."""
    if ms is None:
        return "–"
    if ms >= 1:
        return f"{ms:,.2f} ms"
    if ms >= 1e-3:
        return f"{ms * 1e3:,.2f} µs"
    return f"{ms * 1e6:,.0f} ns"


def latency_bars(rows):
    """Horizontal bars on a log scale: the values span ~6 orders of magnitude (µs core vs ms Python)."""
    vals = [v for _, v, _ in rows if v]
    if len(vals) < 2:
        return ""
    lo, hi = math.log10(min(vals)) - 0.3, math.log10(max(vals))
    out = ""
    for name, v, cls in rows:
        if not v:
            continue
        w = 100 * (math.log10(v) - lo) / (hi - lo) if hi > lo else 100
        out += (f'<div class="bar"><div class="name">{esc(name)}</div><div class="track">'
                f'<div class="fill {cls}" style="width:{max(w, 2):.1f}%"></div></div>'
                f'<div class="nums">{fmt_time(v)}</div></div>')
    return out


def compare_card(b, cpu):
    """Python on CPU vs the FPGA: accuracy and time per image (mlp_test.py + benchmark_cpu.py)."""
    r = b["board"] or {}
    if r.get("kind") != "mlp_board_eval" and not cpu:
        return ""
    c = cpu or {}
    n = r.get("images")
    exact = r.get("bit_exact_images")
    exact_ok = n is not None and exact == n and r.get("vectors_bit_exact", False)
    pill = (("ok", f"bit-exact on {fmt(exact)}/{fmt(n)}") if exact_ok else
            ("bad", f"{fmt(n - exact)} images differ") if n is not None and exact is not None else None)
    kpis = ""
    if r.get("kind") == "mlp_board_eval":
        core_ms, wall_ms = r.get("hw_core_ms"), r.get("hw_wall_ms")
        cpu_ms = [v for v in (c.get("float_single_ms"), c.get("int_single_ms")) if v]
        best_cpu = min(cpu_ms) if cpu_ms else None
        arm = r.get("arm_single_ms")
        kpis = ('<div class="grid kpis">'
                + kpi("FPGA accuracy", fmt(r.get("hw_accuracy"), 2, " %"), f"{fmt(n)} MNIST test images", pill)
                + kpi("Accelerator time / image", fmt_time(core_ms),
                      f"CYCLES {fmt(r.get('cycles_median'))} @ {fmt(r.get('fclk_mhz'))} MHz")
                + kpi("End-to-end / image", fmt_time(wall_ms), "Python + DMA on the ARM")
                + kpi("Accelerator vs ARM CPU", f"{arm / core_ms:,.0f}×" if arm and core_ms else "–",
                      "same board, one image at a time, compute only"
                      + (f"; vs PC CPU {best_cpu / core_ms:,.1f}×" if best_cpu and core_ms else ""))
                + "</div>")
    rows = [  # platform, arithmetic, accuracy, single ms, batched ms, bar class
        ("PC CPU · PyTorch", "float32", c.get("float_accuracy"), c.get("float_single_ms"), c.get("float_batched_ms"), "muted"),
        ("PC CPU · NumPy", f"int{c.get('bits', '')} reference", c.get("int_accuracy"), c.get("int_single_ms"),
         c.get("int_batched_ms"), "muted"),
        ("PYNQ ARM · NumPy", f"int{r.get('bits', '')} reference", r.get("arm_accuracy"), r.get("arm_single_ms"), r.get("arm_batched_ms"), "warn"),
        ("PYNQ FPGA · end-to-end", "hardware", r.get("hw_accuracy"), r.get("hw_wall_ms"), None, "ok"),
        ("PYNQ FPGA · accelerator", "hardware", r.get("hw_accuracy"), r.get("hw_core_ms"), r.get("hw_core_ms"), "ok"),
    ]
    rows = [x for x in rows if x[2] is not None or x[3] is not None]
    table = "".join(f'<tr><td>{esc(p)}</td><td style="text-align:left">{esc(k)}</td><td>{fmt(acc, 2, " %")}</td>'
                    f'<td>{fmt_time(s)}</td><td>{fmt_time(bt)}</td></tr>' for p, k, acc, s, bt, _ in rows)
    warn = ""
    if cpu and r and c.get("shift_policy") and r.get("shift_policy") and c["shift_policy"] != r["shift_policy"]:
        warn = (f'<div class="note"><span class="pill warn">different configs</span> CPU benchmark used shift '
                f'policy {esc(c["shift_policy"])}, the board {esc(r["shift_policy"])}.</div>')
    host = f'PC: {esc(c.get("cpu", "?"))}, {esc(c.get("cores", "?"))} cores' if c else "PC benchmark missing (make cpu-bench)"
    return (f'<h2 style="margin-top:20px">Python vs hardware</h2>{kpis}'
            f'<div class="card" style="margin-bottom:12px"><h2>Accuracy and time per image</h2><div class="scroll"><table>'
            f'<tr><th>Platform</th><th style="text-align:left">Arithmetic</th><th>Accuracy</th>'
            f'<th>One at a time</th><th>Batched</th></tr>{table}</table></div>'
            f'<div class="note">{host}. Integer rows use the same numbers as the hardware, so their accuracy '
            f'must match the FPGA exactly; float32 is the trained model before quantization.</div>{warn}</div>'
            f'<div class="card wide"><h2>Time per image, one at a time (log scale)</h2>'
            f'{latency_bars([(p, s, cls) for p, _, _, s, _, cls in rows])}'
            f'<div class="note">"Accelerator" is CYCLES / clock: the time the hardware itself needs. '
            f'"End-to-end" adds Python, the DMA driver and cache flushes on the ARM, which dominate for single '
            f'images; batching several images per DMA transfer would close that gap.</div></div>')


def board_card(b, cpu=None):
    r = b["board"]
    if (r and r.get("kind") == "mlp_board_eval") or cpu:
        return compare_card(b, cpu)
    if not r:
        return ""
    items = "".join(kpi(k.replace("_", " "), esc(v)) for k, v in r.items() if not isinstance(v, (dict, list)))
    return f'<h2 style="margin-top:20px">On the board</h2><div class="grid kpis">{items}</div>'


def chips(b):
    s, m = b["summary"], b["meta"]
    items = [("core", s["core"]), ("stream", f'{s["stream_w"]} bit' if s["stream_w"] else None),
             ("in → out", f'{s["n_in"]} → {s["n_out"]}' if s["n_in"] else None),
             ("clock", f'{fmt(s["fclk"], 0)} MHz' if s["fclk"] else None),
             ("device", m.get("Device")), ("state", m.get("Design State")), ("built", m.get("Date"))]
    return '<div class="chips">' + "".join(
        f'<span class="chip">{esc(k)} <b>{esc(v)}</b></span>' for k, v in items if v) + "</div>"


def run_section(b, title_tag="h1", cpu=None):
    missing = [f for f in ("utilization.rpt", "timing_summary.rpt", "utilization_hier.rpt") if f not in b["found"]]
    warn = (f'<div class="card" style="margin-bottom:12px"><span class="pill warn">missing</span> '
            f'{esc(", ".join(missing))} in <code>{esc(b["dir"] / "reports")}</code></div>') if missing else ""
    return (f'<section class="run"><{title_tag}>{esc(b["name"])}</{title_tag}>'
            f'<div class="sub">{esc(b["dir"].resolve())}</div>{chips(b)}{warn}'
            f'{timing_cards(b["summary"])}'
            f'<div class="grid two">{resources_card(b)}{breakdown_card(b)}</div>'
            f'{hierarchy_details(b)}{board_card(b, cpu)}</section>')


def compare_table(builds):
    def pct(t):
        return fmt(t[2], 1, "%") if t else "–"
    head = ("<tr><th>Build</th><th>Core</th><th>Clock</th><th>WNS</th><th>Est. Fmax</th><th>LUTs</th>"
            "<th>FFs</th><th>BRAM</th><th>DSP</th><th>Accel LUTs</th><th>Accel BRAM36</th>"
            "<th>Board acc.</th><th>Accel / image</th></tr>")
    rows = ""
    for b in builds:
        s = b["summary"]
        cls = "ok" if s["met"] else "bad" if s["met"] is False else "muted"
        acc = s["accel"] or {}
        rows += (f'<tr><td>{esc(b["name"])}</td><td>{esc(s["core"])}</td><td>{fmt(s["fclk"], 0)} MHz</td>'
                 f'<td><span class="pill {cls}">{fmt(s["wns"], 3)}</span></td><td>{fmt(s["fmax"], 0)}</td>'
                 f'<td>{pct(s["lut"])}</td><td>{pct(s["ff"])}</td><td>{pct(s["bram"])}</td><td>{pct(s["dsp"])}</td>'
                 f'<td>{fmt(acc.get("Total LUTs"))}</td><td>{fmt(acc.get("RAMB36"))}</td>'
                 f'<td>{fmt((b["board"] or {}).get("hw_accuracy"), 2, " %")}</td>'
                 f'<td>{fmt_time((b["board"] or {}).get("hw_core_ms"))}</td></tr>')
    return (f'<div class="card"><h2>Compared builds</h2><div class="scroll"><table>{head}{rows}</table></div>'
            '<div class="note">Percentages are of the whole xc7z020. "Accel" is only our AccelShell + core; '
            'the platform (DMA, interconnect) is the same in every build.</div></div>')


def page(builds, cpu=None):
    title = "Build report" if len(builds) == 1 else "Build comparison"
    body = (run_section(builds[0], cpu=cpu) if len(builds) == 1 else
            f'<h1>{title}</h1><div class="sub">{len(builds)} builds</div><div style="height:16px"></div>'
            + compare_table(builds) + "".join(run_section(b, "h2", cpu) for b in builds))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{CSS}</style></head><body><main>{body}'
            f'<footer>Generated by hw/scripts/report.py on {stamp} from Vivado reports.</footer>'
            f'</main></body></html>')

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Vivado reports -> HTML dashboard")
    ap.add_argument("builds", nargs="*", default=["hw/vivado"], help="build dirs (build.tcl out_dir)")
    ap.add_argument("-o", "--out", help="output html (default: <build>/reports/dashboard.html, "
                                        "or hw/vivado/compare.html for several builds)")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    ap.add_argument("--cpu", default="python/results/cpu_benchmark.json",
                    help="PC CPU benchmark (python -m scripts.benchmark_cpu) for the Python-vs-hardware card")
    a = ap.parse_args()
    cpu_txt = read(a.cpu)
    cpu = json.loads(cpu_txt) if cpu_txt else None

    builds = [load_build(d) for d in a.builds]
    if not any(b["found"] for b in builds):
        sys.exit(f"no Vivado reports found under: {', '.join(a.builds)} (run make bitstream first)")
    out = pathlib.Path(a.out) if a.out else (
        builds[0]["dir"] / "reports" / "dashboard.html" if len(builds) == 1 else pathlib.Path("hw/vivado/compare.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(builds, cpu), encoding="utf-8")
    print(f"[report] {out.resolve()}")
    for b in builds:
        s = b["summary"]
        lut = s["lut"]
        print(f"  {b['name']}: WNS {fmt(s['wns'], 3)} ns, Fmax~{fmt(s['fmax'], 0)} MHz, "
              f"LUT {fmt(lut[2] if lut else None, 1)}%, BRAM {fmt(s['bram'][0] if s['bram'] else None)}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()