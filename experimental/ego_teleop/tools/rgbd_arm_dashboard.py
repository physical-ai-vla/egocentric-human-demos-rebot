"""Build the V0 gate dashboard: N protocol takes -> one self-contained HTML page.

    .venv/bin/python -m ego_teleop.tools.rgbd_arm_dashboard --take outputs/poc/t1 --take outputs/poc/t2 \
        --take outputs/poc/t3 --out outputs/poc/gate.html

A take is a directory containing `v0_palm_pose.parquet` (what `p1_rgbd_arm --stage v0` writes), or the parquet
itself. Metrics come from `rgbd_arm_metrics`; this module only arranges them. The page is static — no JS, no network
at view time — so it can be opened from disk, attached to a message, or published as an Artifact unchanged."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import load_teleop_cfg
from ..transforms.frames import HumanRobotFrameMapper
from . import rgbd_arm_metrics as Mx
from .p1_rgbd_arm import PROTOCOL_60S

LEVEL_LABEL = {"green": "GO", "yellow": "HOLD", "red": "NO-GO", "none": "NOT MEASURED"}
VERDICT_TEXT = {
    "green": "All six gate metrics are green on every take. The first low-speed 3-DoF real-arm test (V2) is permitted.",
    "yellow": "At least one metric is in the yellow band. Watch the trajectory and the overlay video for those "
              "takes before allowing V2 — yellow is not a refusal, it is a request to look.",
    "red": "At least one metric is red on at least one take. V2 stays closed. The worst take decides, because three "
           "takes exist so that one lucky take cannot open the gate.",
    "none": "The gate cannot be evaluated: a required metric was never measured. An unmeasured metric is not a pass.",
}
GATE_TITLE = {"arm_pose_valid_duty": "Arm-pose valid duty", "stationary_xyz_rms_mm": "Static XYZ RMS jitter",
              "stationary_xyz_p95_mm": "Static XYZ p95 deviation", "return_to_start_translation_mm": "Return-to-start error",
              "catastrophic_jump_count": f"Catastrophic jump (>{Mx.JUMP_MM:.0f} mm/frame)",
              "long_tracking_loss_count": f"Long tracking loss (>{Mx.LONG_LOSS_MS:.0f} ms)"}
HEADLINE = ("stationary_xyz_rms_mm", "return_to_start_translation_mm", "arm_pose_valid_duty")
HEADLINE_TITLE = {"stationary_xyz_rms_mm": "Stationary XYZ jitter",
                  "return_to_start_translation_mm": "Return-to-start error",
                  "arm_pose_valid_duty": "Arm-pose valid duty"}


# ---- loading -----------------------------------------------------------------------------------------------------
def load_take(path: str | Path) -> tuple[str, pd.DataFrame]:
    p = Path(path)
    f = p if p.suffix == ".parquet" else p / "v0_palm_pose.parquet"
    if not f.exists(): raise FileNotFoundError(f"no v0_palm_pose.parquet at {p} (run p1_rgbd_arm --stage v0 first)")
    return (p.stem if p.suffix == ".parquet" else p.name), pd.read_parquet(f)


# ---- formatting --------------------------------------------------------------------------------------------------
def fmt(v, unit: str = "", *, dec: int = 1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "—"
    if unit == "%": return f"{v * 100:.1f}%"
    if unit == "" and float(v).is_integer(): return f"{int(v)}"
    return f"{v:.{dec}f}{(' ' + unit) if unit else ''}"


def _svg_trace(df: pd.DataFrame, *, w: int = 900, h: int = 132, pad: int = 26) -> str:
    """Palm XYZ deviation from the take mean, with the protocol segment boundaries behind it."""
    s = Mx.seconds(df)
    ok = Mx.tracked(df)
    P = df[["x", "y", "z"]].to_numpy(np.float64)
    if not ok.any(): return '<p class="empty">no tracked frames</p>'
    mu = P[ok].mean(0)
    dev = (P - mu) * 1000.0
    lim = max(float(np.nanmax(np.abs(dev[ok]))), 5.0)
    t_max = float(s[-1]) if len(s) else 60.0
    X = lambda t: pad + (t / max(t_max, 1e-6)) * (w - 2 * pad)
    Y = lambda v: h / 2 - (v / lim) * (h / 2 - 10)
    out = [f'<svg viewBox="0 0 {w} {h}" class="trace" role="img" aria-label="palm XYZ deviation over the take">']
    for i, (name, a, b, _) in enumerate(PROTOCOL_60S):
        if a > t_max: break
        out.append(f'<rect x="{X(a):.1f}" y="0" width="{max(X(min(b, t_max)) - X(a), 0):.1f}" height="{h}" '
                   f'class="seg{i % 2}"/>')
        out.append(f'<text x="{X(a) + 3:.1f}" y="11" class="segtxt">{html.escape(name)}</text>')
    out.append(f'<line x1="{pad}" y1="{Y(0):.1f}" x2="{w - pad}" y2="{Y(0):.1f}" class="axis"/>')
    for v in (-lim, lim):
        out.append(f'<text x="2" y="{Y(v) + (10 if v < 0 else 4):.1f}" class="tick">{v:+.0f}</text>')
    for i, (axis, cls) in enumerate((("x", "ax"), ("y", "ay"), ("z", "az"))):
        pts, run = [], []
        for j in range(len(s)):
            if ok[j] and np.isfinite(dev[j, i]): run.append(f"{X(s[j]):.1f},{Y(dev[j, i]):.1f}")
            elif run: pts.append(run); run = []
        if run: pts.append(run)
        for seg in pts:
            if len(seg) > 1: out.append(f'<polyline points="{" ".join(seg)}" class="{cls}"/>')
    out.append(f'<text x="{w - pad}" y="{h - 3}" class="tick" text-anchor="end">mm, deviation from take mean</text>')
    out.append("</svg>")
    return "".join(out)


def _svg_ribbon(df: pd.DataFrame, *, w: int = 900, h: int = 16, pad: int = 26) -> str:
    """Arm-pose health as a time ribbon: one 1.5 s outage is visible here and invisible in a duty percentage."""
    s = Mx.seconds(df)
    if not len(s): return ""
    t_max = float(s[-1]) or 60.0
    X = lambda t: pad + (t / t_max) * (w - 2 * pad)
    cls = {Mx.ARM_OK: "hok", Mx.ARM_DEGRADED: "hdeg", Mx.ARM_LOST: "hlost"}
    out = [f'<svg viewBox="0 0 {w} {h}" class="ribbon" role="img" aria-label="arm pose health over the take">']
    hs = df["arm_pose_health"].to_numpy()
    i = 0
    while i < len(hs):
        j = i
        while j + 1 < len(hs) and hs[j + 1] == hs[i]: j += 1
        x0, x1 = X(s[i]), X(s[min(j + 1, len(s) - 1)])
        out.append(f'<rect x="{x0:.1f}" y="0" width="{max(x1 - x0, 0.6):.1f}" height="{h}" class="{cls.get(hs[i], "hlost")}"/>')
        i = j + 1
    out.append("</svg>")
    return "".join(out)


# ---- page --------------------------------------------------------------------------------------------------------
def render(takes: list[tuple[str, pd.DataFrame, dict]], agg: dict, *, cfg_note: str, mapper_desc: dict) -> str:
    v = agg["verdict"]
    n = len(takes)
    head_tiles = "".join(
        f'<div class="tile l-{agg["gate"][k]["level"]}"><p class="tk">{HEADLINE_TITLE[k]}</p>'
        f'<p class="tv">{fmt(agg["gate"][k]["worst_value"] if agg["gate"][k]["sense"] == "max" else agg["gate"][k]["best"], agg["gate"][k]["unit"])}</p>'
        f'<p class="tm">worst of {n} · gate {"≥" if agg["gate"][k]["sense"] == "min" else "≤"} '
        f'{fmt(agg["gate"][k]["green"], agg["gate"][k]["unit"])}</p></div>'
        for k in HEADLINE)

    rows = []
    for key, unit, green, yellow, sense in Mx.GATE:
        g = agg["gate"][key]
        cells = "".join(f'<td class="num l-{t[2]["gate"][key]["level"]}">{fmt(t[2]["gate"][key]["value"], unit)}</td>'
                        for t in takes)
        op = "≥" if sense == "min" else "≤"
        rows.append(f'<tr><th scope="row"><span class="stripe s-{g["level"]}"></span>{GATE_TITLE[key]}</th>'
                    f'{cells}<td class="num thr">{op} {fmt(green, unit)}</td>'
                    f'<td class="num thr dim">{op} {fmt(yellow, unit)}</td>'
                    f'<td><span class="pill p-{g["level"]}">{LEVEL_LABEL[g["level"]]}</span></td></tr>')
    gate_head = "".join(f'<th scope="col" class="num">{html.escape(t[0])}</th>' for t in takes)

    seg_rows = []
    for name, a, b, hint in PROTOCOL_60S:
        cs = []
        for _, _, m in takes:
            s = m["segments"].get(name)
            if not s: cs.append('<td class="num">—</td>' * 8); continue
            d, j = s["displacement"], s["jitter"]
            cam = d.get("camera_mm", {}); rob = d.get("robot_mm", {})
            # a window that moved is marked, not silently reported as noise
            mv = "" if (j["still"] or not np.isfinite(j["rms_mm"])) else ' <span class="mv">mv</span>'
            cam_s = " ".join(fmt(cam.get(x), dec=0) for x in "xyz")
            rob_s = " ".join(fmt(rob.get(x), dec=0) for x in "xyz")
            cs.append(
                f'<td class="num">{fmt(s["arm_pose_valid_duty"], "%")}</td>'
                f'<td class="num mono">{cam_s}</td>'
                f'<td class="num mono">{rob_s}</td>'
                f'<td class="num">{fmt(d.get("rotation_deg"), "°")}</td>'
                f'<td class="num">{fmt(j["rms_mm"], "mm")}{mv}</td>'
                f'<td class="num">{fmt(j.get("orientation_p95_deg"), "°")}</td>'
                f'<td class="num">{fmt(s["jumps"])}</td>'
                f'<td class="num">{fmt(s["arm_pose_lost_run_ms"]["p95"], "ms", dec=0)}</td>')
        seg_rows.append(f'<tr><th scope="row"><span class="secs">{a}–{b}s</span>{html.escape(name)}'
                        f'<span class="hint">{html.escape(hint)}</span></th>{"".join(cs)}</tr>')

    leak_cards = []
    for label, _, m in takes:
        lk = (m["segments"].get("palm_rotation_only") or {}).get("translation_leak")
        if not lk or not np.isfinite(lk.get("p95_mm", np.nan)): continue
        bad = np.isfinite(lk.get("mm_per_10deg", np.nan)) and lk["mm_per_10deg"] > 5.0
        leak_cards.append(
            f'<div class="leak{" leak-bad" if bad else ""}"><p class="lk">{html.escape(label)}</p>'
            f'<p class="lv">{fmt(lk.get("mm_per_10deg"), dec=1)}<span class="lu">mm / 10°</span></p>'
            f'<p class="lm">{fmt(lk.get("p95_mm"), "mm")} p95 excursion under {fmt(lk.get("rotation_deg"), "°", dec=0)} '
            f'of wrist rotation · net {fmt(lk.get("net_mm"), "mm")}</p></div>')

    per_take = []
    for label, df, m in takes:
        lr = m["arm_pose_lost_run_ms"]; tr = m["tracking_lost_run_ms"]
        per_take.append(f"""<section class="take">
<h3>{html.escape(label)}<span class="tmeta">{m["duration_s"]:.1f}s · {m["frames"]} frames · duty {fmt(m["arm_pose_valid_duty"], "%")}
 (usable {fmt(m["arm_pose_usable_duty"], "%")})</span></h3>
{_svg_trace(df)}
{_svg_ribbon(df)}
<p class="legend"><span class="k ax"></span>x<span class="k ay"></span>y<span class="k az"></span>z
<span class="sp"></span><span class="k hok"></span>ARM_POSE_OK<span class="k hdeg"></span>DEGRADED<span class="k hlost"></span>LOST</p>
<dl class="kv">
<dt>arm-pose lost runs</dt><dd>{lr["n"]} · median {fmt(lr["median"], "ms", dec=0)} · p95 {fmt(lr["p95"], "ms", dec=0)} · max {fmt(lr["max"], "ms", dec=0)}</dd>
<dt>tracking lost runs</dt><dd>{tr["n"]} · median {fmt(tr["median"], "ms", dec=0)} · p95 {fmt(tr["p95"], "ms", dec=0)} · max {fmt(tr["max"], "ms", dec=0)}</dd>
<dt>max frame step</dt><dd>{fmt(m["max_frame_step_mm"], "mm")} · {m["catastrophic_jump_count"]} over {Mx.JUMP_MM:.0f} mm</dd>
<dt>stationary window</dt><dd>{fmt(m["stationary"]["rms_mm"], "mm")} RMS · {fmt(m["stationary"]["p95_mm"], "mm")} p95 ·
 drift {fmt(m["stationary"]["drift_mm"], "mm")} · {"stationary" if m["stationary"]["still"] else "MOVED — jitter withheld"}</dd>
<dt>return to start</dt><dd>{fmt(m["return_to_start_translation_mm"], "mm")} ·
 {fmt(m["return_to_start_rotation_deg"], "°")}{" · " + html.escape(m["return_to_start"].get("note", "")) if m["return_to_start"].get("note") else ""}</dd>
<dt>top reasons</dt><dd class="mono">{html.escape(", ".join(f"{k} ×{n}" for k, n in list(m["reasons"].items())[:4]) or "—")}</dd>
</dl></section>""")

    return f"""<title>RGB-D Arm Gate</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
:root {{
  --ground:#eef2f3; --surface:#ffffff; --ink:#0f1a1e; --muted:#5b7078; --line:#d3dcdf; --accent:#0b6e7f;
  --pass:#1a7f4b; --hold:#a06800; --fail:#b4231c; --none:#7a8a90;
  --passbg:#e6f3ec; --holdbg:#fbf1de; --failbg:#fbe9e7; --nonebg:#eceff0;
  --x:#0b6e7f; --y:#a0562a; --z:#5b4b9e;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --cond:"IBM Plex Sans Condensed","IBM Plex Sans",system-ui,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace;
}}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
  --ground:#0b1215; --surface:#121c20; --ink:#e4edf0; --muted:#8ba1a9; --line:#233238; --accent:#3ab3c4;
  --pass:#4cc38a; --hold:#e0a82e; --fail:#ff6f61; --none:#7a8a90;
  --passbg:#132a20; --holdbg:#2c2413; --failbg:#2e1a18; --nonebg:#1a2428;
  --x:#3ab3c4; --y:#d99a5b; --z:#9a8ce0;
}} }}
:root[data-theme="dark"] {{
  --ground:#0b1215; --surface:#121c20; --ink:#e4edf0; --muted:#8ba1a9; --line:#233238; --accent:#3ab3c4;
  --pass:#4cc38a; --hold:#e0a82e; --fail:#ff6f61; --none:#7a8a90;
  --passbg:#132a20; --holdbg:#2c2413; --failbg:#2e1a18; --nonebg:#1a2428;
  --x:#3ab3c4; --y:#d99a5b; --z:#9a8ce0;
}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:15px;line-height:1.55;
  padding-block:32px;padding-left:20px;padding-right:20px;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1140px;margin:0 auto;display:flex;flex-direction:column;gap:28px}}
h1,h2,h3{{font-family:var(--cond);font-weight:700;text-wrap:balance;margin:0;letter-spacing:-.01em}}
h1{{font-size:2.05rem;line-height:1.1}} h2{{font-size:1.16rem}} h3{{font-size:1.02rem}}
p{{margin:0}}
.eyebrow{{font-family:var(--mono);font-size:.7rem;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}}
header{{display:flex;flex-direction:column;gap:10px;border-bottom:2px solid var(--ink);padding-bottom:18px}}
.sub{{color:var(--muted);max-width:66ch}}
.prov{{font-family:var(--mono);font-size:.74rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 14px}}
.verdict{{display:flex;gap:14px;align-items:flex-start;padding:14px 16px;border-radius:3px;
  border-left:4px solid var(--lv);background:var(--lvbg)}}
.verdict .big{{font-family:var(--cond);font-weight:700;font-size:1.3rem;color:var(--lv);white-space:nowrap}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}
.tile{{background:var(--surface);border:1px solid var(--line);border-top:3px solid var(--lv);padding:14px 16px}}
.tk{{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}
.tv{{font-family:var(--cond);font-weight:700;font-size:2.3rem;line-height:1.1;font-variant-numeric:tabular-nums;color:var(--lv)}}
.tm{{font-size:.8rem;color:var(--muted)}}
.l-green{{--lv:var(--pass);--lvbg:var(--passbg)}} .l-yellow{{--lv:var(--hold);--lvbg:var(--holdbg)}}
.l-red{{--lv:var(--fail);--lvbg:var(--failbg)}} .l-none{{--lv:var(--none);--lvbg:var(--nonebg)}}
.scroll{{overflow-x:auto;background:var(--surface);border:1px solid var(--line)}}
table{{border-collapse:collapse;width:100%;font-size:.88rem}}
th,td{{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:baseline}}
thead th{{font-family:var(--mono);font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  border-bottom:1px solid var(--ink);white-space:nowrap}}
tbody tr:last-child td,tbody tr:last-child th{{border-bottom:none}}
.num{{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}}
td.num.l-green{{color:var(--pass)}} td.num.l-yellow{{color:var(--hold)}} td.num.l-red{{color:var(--fail);font-weight:600}}
td.num.l-none{{color:var(--none)}}
.thr{{color:var(--muted);font-size:.8rem}} .dim{{opacity:.62}}
.stripe{{display:inline-block;width:3px;height:1em;margin-right:8px;vertical-align:-.15em;background:var(--lv)}}
.s-green{{--lv:var(--pass)}} .s-yellow{{--lv:var(--hold)}} .s-red{{--lv:var(--fail)}} .s-none{{--lv:var(--none)}}
.pill{{font-family:var(--mono);font-size:.68rem;letter-spacing:.07em;padding:2px 7px;border-radius:2px;white-space:nowrap}}
.p-green{{background:var(--passbg);color:var(--pass)}} .p-yellow{{background:var(--holdbg);color:var(--hold)}}
.p-red{{background:var(--failbg);color:var(--fail)}} .p-none{{background:var(--nonebg);color:var(--none)}}
.secs{{font-family:var(--mono);font-size:.72rem;color:var(--accent);margin-right:9px}}
.hint{{display:block;font-size:.76rem;color:var(--muted)}}
.mv{{font-family:var(--mono);font-size:.66rem;color:var(--hold)}}
.leaks{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}}
.leak{{background:var(--surface);border:1px solid var(--line);padding:13px 15px;border-left:3px solid var(--accent)}}
.leak-bad{{border-left-color:var(--fail)}}
.lk{{font-family:var(--mono);font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}}
.lv{{font-family:var(--cond);font-weight:700;font-size:1.95rem;font-variant-numeric:tabular-nums;line-height:1.15}}
.lu{{font-family:var(--mono);font-size:.74rem;font-weight:400;color:var(--muted);margin-left:7px}}
.lm{{font-size:.8rem;color:var(--muted)}}
.take{{background:var(--surface);border:1px solid var(--line);padding:16px 18px;display:flex;flex-direction:column;gap:9px}}
.tmeta{{font-family:var(--mono);font-size:.74rem;font-weight:400;color:var(--muted);margin-left:11px}}
svg.trace,svg.ribbon{{width:100%;height:auto;display:block}}
.seg0{{fill:transparent}} .seg1{{fill:var(--line);opacity:.32}}
.segtxt{{font-family:var(--mono);font-size:8.5px;fill:var(--muted)}}
.tick{{font-family:var(--mono);font-size:8.5px;fill:var(--muted)}}
.axis{{stroke:var(--muted);stroke-width:.6;opacity:.55}}
polyline{{fill:none;stroke-width:1.2}} .ax{{stroke:var(--x)}} .ay{{stroke:var(--y)}} .az{{stroke:var(--z)}}
rect.hok{{fill:var(--pass)}} rect.hdeg{{fill:var(--hold)}} rect.hlost{{fill:var(--fail)}}
.legend{{font-family:var(--mono);font-size:.7rem;color:var(--muted);display:flex;align-items:center;gap:5px;flex-wrap:wrap}}
.k{{width:11px;height:3px;display:inline-block;margin-left:9px}} .k.ax{{background:var(--x)}} .k.ay{{background:var(--y)}}
.k.az{{background:var(--z)}} .k.hok{{background:var(--pass);height:9px}} .k.hdeg{{background:var(--hold);height:9px}}
.k.hlost{{background:var(--fail);height:9px}} .sp{{width:18px;display:inline-block}}
.kv{{display:grid;grid-template-columns:minmax(140px,auto) 1fr;gap:4px 16px;margin:2px 0 0;font-size:.85rem}}
dt{{font-family:var(--mono);font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding-top:2px}}
dd{{margin:0;font-variant-numeric:tabular-nums}}
.mono{{font-family:var(--mono);font-size:.8rem}}
.notes{{border-top:1px solid var(--line);padding-top:16px;display:flex;flex-direction:column;gap:7px;
  font-size:.83rem;color:var(--muted);max-width:78ch}}
.notes b{{color:var(--ink);font-weight:500}}
.empty{{color:var(--muted);font-family:var(--mono);font-size:.8rem;padding:10px 0}}
@media (max-width:560px) {{ h1{{font-size:1.6rem}} .tv{{font-size:1.9rem}} .kv{{grid-template-columns:1fr}} }}
</style>
<div class="wrap">
<header>
  <p class="eyebrow">Vision-only RGB-D arm POC · V0 gate</p>
  <h1>Can one fixed RGB-D camera carry a reBot arm?</h1>
  <p class="sub">{n} 60-second protocol take{"s" if n != 1 else ""} through the same pose source the analysis uses.
  Six metrics decide whether the first low-speed 3-DoF real-arm test is allowed. This is an engineering gate for
  hardware permission, not a paper benchmark.{
  " <b>Fewer than three takes: the gate is not decidable yet</b> — three exist so one lucky take cannot open it."
  if n < 3 else ""}</p>
  <p class="prov"><span>pose_source: fixed_rgbd_hand</span><span>camera_mode: fixed</span><span>imu_used: false</span>
  <span>causal: true</span><span>{n} take{"s" if n != 1 else ""}</span><span>{html.escape(cfg_note)}</span></p>
</header>

<div class="verdict l-{v}"><p class="big">{LEVEL_LABEL[v]}</p><p>{VERDICT_TEXT[v]}</p></div>

<section>
  <h2>The three numbers</h2>
  <div class="tiles">{head_tiles}</div>
</section>

<section>
  <h2>Gate matrix</h2>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Metric</th>{gate_head}<th scope="col" class="num">green</th>
    <th scope="col" class="num">yellow</th><th scope="col">worst</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table></div>
</section>

<section>
  <h2>Per segment</h2>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Segment</th>{"".join(f'<th scope="col" class="num">duty</th><th scope="col" class="num">Δ camera xyz</th><th scope="col" class="num">Δ robot xyz</th><th scope="col" class="num">Δ rot</th><th scope="col" class="num">jitter</th><th scope="col" class="num">ori p95</th><th scope="col" class="num">jumps</th><th scope="col" class="num">lost p95</th>' for _ in takes)}</tr></thead>
    <tbody>{"".join(seg_rows)}</tbody>
  </table></div>
  <p class="notes" style="border:none;padding-top:10px">Δ camera and Δ robot are the same motion in millimetres,
  before and after <b>HumanRobotFrameMapper</b> — read the axis map off this pair. Mapping under test:
  <span class="mono">{html.escape(json.dumps(mapper_desc))}</span></p>
</section>

{f'''<section>
  <h2>Translation leak during wrist-only rotation</h2>
  <p class="sub">The operator rotated the wrist in place. Whatever the palm origin did instead of staying put is
  leak — and it only means something against the rotation that caused it. Above roughly 5&nbsp;mm per 10°, the palm
  origin definition needs revisiting before any 6-DoF work.</p>
  <div class="leaks">{"".join(leak_cards)}</div>
</section>''' if leak_cards else ''}

<section style="display:flex;flex-direction:column;gap:16px">
  <h2>Per take</h2>
  {"".join(per_take)}
</section>

<div class="notes">
  <p><b>A run is measured to the next good frame.</b> A 10-frame outage at 30&nbsp;Hz is 333&nbsp;ms of robot-side
  hold, not 300 — the arm target is frozen until a pose arrives. Runs still open at the end of a take are counted.</p>
  <p><b>A jump is never measured across a gap.</b> A hand that disappears in one place and reappears in another is a
  tracking loss, already counted as one; charging it again as motion would make this gate unreadable.</p>
  <p><b>Jitter is withheld where the hand moved.</b> Deviation about a window mean is a noise measure only if the
  window holds no motion. Windows whose net drift exceeds {Mx.STILL_DRIFT_MM:.0f}&nbsp;mm are marked
  <span class="mv">mv</span> and excluded from the gate rather than reported as noise.</p>
  <p><b>Unmeasured is never a pass.</b> A metric with no data reads NOT MEASURED and blocks the gate.</p>
  <p><b>The worst take decides.</b> Three takes exist so that one lucky take cannot open the gate.</p>
</div>
</div>"""


def build(take_paths: list[str], out: Path, *, segments: dict | None = None) -> dict:
    cfg = load_teleop_cfg()
    poc = cfg.rgbd_arm_poc
    mapper = HumanRobotFrameMapper(poc.frames)
    segs = segments or {n: (float(a), float(b)) for n, a, b, _ in PROTOCOL_60S}
    takes = []
    for p in take_paths:
        label, df = load_take(p)
        takes.append((label, df, Mx.take_metrics(df, segs, mapper=mapper, name=label)))
    agg = Mx.aggregate([t[2] for t in takes])
    note = f"scale {list(poc.frames.translation_scale)} · palm {poc.palm.origin}/{poc.palm.orientation}"
    out.write_text(render(takes, agg, cfg_note=note, mapper_desc=mapper.describe()))
    return dict(verdict=agg["verdict"], gate={k: g["level"] for k, g in agg["gate"].items()},
                takes={t[0]: t[2]["gate"] for t in takes}, html=str(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V0 gate dashboard over one or more protocol takes")
    ap.add_argument("--take", action="append", required=True, metavar="DIR_OR_PARQUET",
                    help="directory holding v0_palm_pose.parquet (repeatable; 3 takes expected)")
    ap.add_argument("--segment", action="append", metavar="NAME:T0:T1", help="override the protocol windows")
    ap.add_argument("--out", default="outputs/rgbd_arm_poc/gate.html")
    ap.add_argument("--json", default=None, help="also write the metric summary as JSON")
    a = ap.parse_args(argv)
    segs = None
    if a.segment:
        segs = {}
        for spec in a.segment:
            n, t0, t1 = spec.split(":"); segs[n] = (float(t0), float(t1))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    rep = build(a.take, out, segments=segs)
    if a.json: Path(a.json).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    print(f"\nVERDICT: {LEVEL_LABEL[rep['verdict']]}   ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
