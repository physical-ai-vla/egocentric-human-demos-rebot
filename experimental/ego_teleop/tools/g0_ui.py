"""G0-0 Mount Qualification — one local page to aim, record, analyze and read the verdict.

    .venv/bin/python -m ego_teleop.tools.g0_ui [--port 8720] [--open] [--mock]

The page drives `g0_mount_qual` as SUBPROCESSES (preview / record / analyze), never in-process: the take must not
share a process with MediaPipe (frame drops), and preview and record both need the camera, so the server makes them
mutually exclusive — starting a take stops the preview first. Served over HTTP like the other HUDs (no cv2 window).
Nothing here talks to the robot."""
from __future__ import annotations
import argparse
import collections
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import g0_mount_qual as G

REPO = Path(__file__).resolve().parents[2]
RUN_DIR = REPO / "runs" / "g0_ui"
DEFAULT_ROOT = REPO / "datasets" / "human_handumi_raw" / "G0MOUNT"
SEG_RE = re.compile(r"^\s{2}(\w+)\s+([\d.]+)s\s+(.*)$")
EP_RE = re.compile(r"^episode:\s+(\S+)")


class Job:
    """One subprocess with its stdout tail."""

    def __init__(self, kind: str, argv: list[str]) -> None:
        self.kind, self.argv, self.t0 = kind, argv, time.time()
        self.log: collections.deque[str] = collections.deque(maxlen=400)
        self.proc = subprocess.Popen(argv, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.segment: dict | None = None; self.segments_done = 0; self.episode: str | None = None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if re.match(r"^[WI]\d{4} ", line) or "inference_feedback" in line or "gl_context" in line: continue   # MediaPipe noise
            self.log.append(line)
            m = SEG_RE.match(line)
            if m and self.kind == "record":
                if self.segment: self.segments_done += 1
                self.segment = dict(name=m.group(1), dur_s=float(m.group(2)), cue=m.group(3), t0=time.time())
            e = EP_RE.match(line)
            if e: self.episode = e.group(1)

    @property
    def running(self) -> bool: return self.proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if self.running:
            self.proc.terminate()
            try: self.proc.wait(timeout)
            except subprocess.TimeoutExpired: self.proc.kill()

    def state(self) -> dict:
        return dict(kind=self.kind, running=self.running, rc=self.proc.poll(), elapsed=time.time() - self.t0,
                    segment=self.segment, segments_done=self.segments_done, episode=self.episode, log=list(self.log)[-60:])


class App:
    def __init__(self, *, root: Path, hardware: str, mock: bool) -> None:
        self.root, self.hardware, self.mock = root, hardware, mock
        self.preview: Job | None = None; self.job: Job | None = None; self.lock = threading.Lock()
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    def _py(self, *args: str) -> list[str]:
        return [sys.executable, "-u", "-m", "ego_teleop.tools.g0_mount_qual", *args]

    # -- actions ---------------------------------------------------------------------------------------------
    def start_preview(self) -> str:
        with self.lock:
            if self.job and self.job.running and self.job.kind == "record": return "recording — preview not allowed"
            if self.preview and self.preview.running: return "preview already running"
            a = ["preview", "--hardware", self.hardware, "--jpeg", str(RUN_DIR / "preview.jpg"), "--status", str(RUN_DIR / "preview.json")]
            if self.mock: a.append("--mock")
            (RUN_DIR / "preview.json").unlink(missing_ok=True)
            self.preview = Job("preview", self._py(*a)); return "preview started"

    def stop_preview(self) -> str:
        with self.lock:
            if self.preview: self.preview.stop(); return "preview stopped"
            return "no preview"

    def start_record(self, notes: str) -> str:
        with self.lock:
            if self.job and self.job.running: return f"{self.job.kind} already running"
            if self.preview and self.preview.running:
                self.preview.stop(); time.sleep(1.5)            # release the camera before the take opens it
            a = ["record", "--hardware", self.hardware, "--root", str(self.root)]
            if notes: a += ["--notes", notes]
            if self.mock: a += ["--mock", "--time-scale", "0.1"]
            self.job = Job("record", self._py(*a)); return "recording started"

    def abort(self) -> str:
        with self.lock:
            if self.job and self.job.running: self.job.stop(); return f"{self.job.kind} aborted"
            return "nothing running"

    def start_analyze(self, episode: str, views: str) -> str:
        with self.lock:
            if self.job and self.job.running: return f"{self.job.kind} already running"
            if not (Path(episode) / "episode_meta.json").exists(): return f"not an episode: {episode}"
            self.job = Job("analyze", self._py("analyze", episode, "--views", views or "raw,rect")); return "analysis started"

    # -- views -----------------------------------------------------------------------------------------------
    def episodes(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("G0MOUNT_*/episode_*"), key=lambda p: p.stat().st_mtime, reverse=True)[:30]:
            rep = p / "derived" / "g0_mount" / "g0_report.json"; verdict = None
            if rep.exists():
                try: verdict = json.loads(rep.read_text()).get("verdict")
                except Exception: verdict = "unreadable report"
            try: meta = json.loads((p / "episode_meta.json").read_text())
            except Exception: meta = {}
            out.append(dict(path=str(p), name=f"{p.parent.name}/{p.name}", verdict=verdict, notes=meta.get("notes", ""),
                            duration_s=meta.get("duration_s"), wall=meta.get("t_start_wall_iso", "")))
        return out

    def state(self) -> dict:
        pv = None
        if self.preview and self.preview.running:
            try: pv = json.loads((RUN_DIR / "preview.json").read_text())
            except Exception: pv = None
        return dict(hardware=self.hardware, mock=self.mock, root=str(self.root),
                    preview=dict(running=bool(self.preview and self.preview.running), status=pv,
                                 log=list(self.preview.log)[-8:] if self.preview else []),
                    job=self.job.state() if self.job else None, episodes=self.episodes(),
                    protocol=[dict(name=n, dur_s=d, cue=c) for n, d, c in G.PROTOCOL], gates=G.GATES)


def _nan_to_none(o):
    if isinstance(o, float) and o != o: return None
    if isinstance(o, dict): return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list): return [_nan_to_none(v) for v in o]
    return o


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(_nan_to_none(obj), default=str).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path == "/": return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if u.path == "/state": return self._json(app.state())
            if u.path == "/preview.jpg":
                p = RUN_DIR / "preview.jpg"
                return self._send(200, p.read_bytes(), "image/jpeg") if p.exists() else self._send(404, b"", "text/plain")
            if u.path in ("/report", "/file"):
                ep = Path(q.get("ep", [""])[0]).resolve()
                if not (ep.is_relative_to(app.root.resolve()) or ep.is_relative_to(REPO)): return self._send(403, b"outside dataset", "text/plain")
                d = ep / "derived" / "g0_mount"
                if u.path == "/report":
                    p = d / "g0_report.json"
                    if not p.exists(): return self._json({"missing": True}, 404)
                    # the report is written by Python json, which emits NaN; browsers' JSON.parse rejects it
                    return self._send(200, json.dumps(_nan_to_none(json.loads(p.read_text()))).encode(), "application/json")
                name = q.get("name", [""])[0]
                if name not in ("overlay.jpg", "signals.png", "mask_vi.png", "texture_vi.png"): return self._send(400, b"", "text/plain")
                p = d / name
                return self._send(200, p.read_bytes(), "image/png" if name.endswith(".png") else "image/jpeg") if p.exists() else self._send(404, b"", "text/plain")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            u = urlparse(self.path); n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            acts = {"/preview/start": lambda: app.start_preview(), "/preview/stop": lambda: app.stop_preview(),
                    "/record": lambda: app.start_record(body.get("notes", "")), "/abort": lambda: app.abort(),
                    "/analyze": lambda: app.start_analyze(body.get("episode", ""), body.get("views", "raw,rect"))}
            if u.path not in acts: return self._send(404, b"", "text/plain")
            self._json(dict(msg=acts[u.path]()))
    return H


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G0-0 Mount Qualification</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d232b;--mut:#6a7380;--line:#e1e5ea;--acc:#2f6fdf;--ok:#1f9d55;--bad:#d64545;--warn:#c98a00}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--fg:#e6e9ee;--mut:#9aa3ae;--line:#2c323b;--acc:#6aa0ff;--ok:#41c27b;--bad:#ff6b6b;--warn:#e0b040}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
h1{font-size:17px;margin:0}.mut{color:var(--mut)}main{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;padding:16px 20px}
@media (max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.card h2{font-size:14px;margin:0 0 10px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:7px;padding:7px 12px;cursor:pointer}
button.pri{background:var(--acc);border-color:var(--acc);color:#fff}button.danger{color:var(--bad);border-color:var(--bad)}button:disabled{opacity:.45;cursor:default}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}img{max-width:100%;border-radius:6px;display:block}
.kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-variant-numeric:tabular-nums}.kv span:nth-child(odd){color:var(--mut)}
.cue{font-size:34px;font-weight:700;margin:6px 0}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>div{height:100%;background:var(--acc)}
.tl{display:flex;height:22px;border-radius:5px;overflow:hidden;margin-top:8px}.tl div{border-right:1px solid var(--card);font-size:10px;color:#fff;overflow:hidden;white-space:nowrap;padding:3px 2px}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;max-height:200px;overflow:auto;font-size:11.5px;margin:8px 0 0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;font-size:12.5px}th{color:var(--mut);font-weight:500}
.pass{color:var(--ok);font-weight:600}.fail{color:var(--bad);font-weight:600}.nm{color:var(--warn);font-weight:600}
.verdict{font-size:16px;font-weight:700;padding:10px 12px;border-radius:8px;border:1px solid var(--line);margin-bottom:10px}
.eps{max-height:220px;overflow:auto}.ep{padding:6px 8px;border-radius:6px;cursor:pointer;display:flex;justify-content:space-between;gap:8px}.ep:hover,.ep.sel{background:var(--bg)}
input[type=text]{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);min-width:220px}
.lamp{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;background:var(--mut)}.lamp.on{background:var(--ok)}.lamp.off{background:var(--bad)}
.imgs{display:grid;grid-template-columns:1fr;gap:10px;margin-top:10px}.fails{margin:4px 0 0 0;padding-left:18px;color:var(--mut);font-size:12.5px}
</style></head><body>
<header><h1>G0-0 Wrist Mount Qualification</h1><span class="mut" id="meta"></span></header>
<main>
<section style="display:grid;gap:16px;align-content:start">
  <div class="card"><h2>1 · 마운트 조준 (미리보기)</h2>
    <div class="row"><button class="pri" id="pvStart">미리보기 시작</button><button id="pvStop">정지</button>
      <span class="mut">엄지·검지 끝이 초록 점으로 보이고, 손 옆으로 환경이 넉넉히 남게 조정하세요.</span></div>
    <div style="margin-top:10px"><img id="pv" alt="" style="display:none"></div>
    <div class="kv" id="pvkv" style="margin-top:8px"></div></div>
  <div class="card"><h2>2 · 녹화 (약 100초 프로토콜)</h2>
    <div class="row"><input type="text" id="notes" placeholder="메모 (예: 35° 위쪽, v1 브래킷)"><button class="pri" id="rec">녹화 시작</button><button class="danger" id="abort">중단</button></div>
    <div id="recbox" style="margin-top:10px"><div class="mut">녹화 시작 시 미리보기는 자동으로 꺼집니다. 노출은 미리 고정하세요.</div></div>
    <div class="tl" id="tl"></div></div>
  <div class="card"><h2>로그</h2><div class="mut" id="jobline"></div><pre id="log"></pre></div>
</section>
<section style="display:grid;gap:16px;align-content:start">
  <div class="card"><h2>3 · 에피소드 / 분석</h2>
    <div class="eps" id="eps"></div>
    <div class="row" style="margin-top:8px"><input type="text" id="epPath" placeholder="에피소드 경로" style="flex:1">
      <label class="mut"><input type="checkbox" id="rectView" checked> rect view 포함</label><button class="pri" id="ana">분석</button></div></div>
  <div class="card" id="repCard"><h2>4 · 결과</h2><div id="rep" class="mut">에피소드를 선택하세요.</div></div>
</section></main>
<script>
const $=id=>document.getElementById(id); let ST=null, sel=null, shownRep=null, lastJobRc=null;
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json()).then(j=>{$('jobline').textContent=j.msg});
const f=(x,d=2)=>x==null||Number.isNaN(x)?'–':(typeof x==='number'?x.toFixed(d):x);
const cls=v=>v===true?'pass':v===false?'fail':'nm', word=v=>v===true?'PASS':v===false?'FAIL':'NOT MEASURED';
const COLORS={still_open:'#7a8899',open:'#3f9d5b',half:'#c9a227',pinch:'#cf5151',pinch_fast:'#a5486e',tx:'#3b74c9',ty:'#3b74c9',tz:'#3b74c9',roll:'#6b55c4',pitch:'#6b55c4',yaw:'#6b55c4',natural:'#2a8a8a',return_still:'#7a8899'};
$('pvStart').onclick=()=>post('/preview/start'); $('pvStop').onclick=()=>post('/preview/stop');
$('rec').onclick=()=>post('/record',{notes:$('notes').value}); $('abort').onclick=()=>post('/abort');
$('ana').onclick=()=>{const ep=$('epPath').value.trim(); if(ep) post('/analyze',{episode:ep,views:$('rectView').checked?'raw,rect':'raw'})};
function timeline(p,job){const tot=p.reduce((a,s)=>a+s.dur_s,0); const done=job&&job.kind==='record'?job.segments_done:-1;
  $('tl').innerHTML=p.map((s,i)=>`<div title="${s.name} ${s.dur_s}s ${s.cue}" style="width:${100*s.dur_s/tot}%;background:${COLORS[s.name]||'#888'};opacity:${done<0||i<=done?1:.35}">${s.name}</div>`).join('')}
function recBox(job,p){ if(!job||job.kind!=='record'){return}
  const s=job.segment; if(!job.running){$('recbox').innerHTML=job.rc===0?`<div class="pass">녹화 완료</div><div class="mut">${job.episode||''}</div>`:`<div class="fail">녹화 종료 (rc ${job.rc})</div>`;return}
  if(!s){$('recbox').innerHTML='<div class="cue">준비…</div><div class="mut">장치 연결 중</div>';return}
  const el=(Date.now()/1000-s.t0), frac=Math.min(1,el/s.dur_s), nxt=p[job.segments_done+1];
  $('recbox').innerHTML=`<div class="mut">${job.segments_done+1} / ${p.length} · ${s.name}</div><div class="cue">${s.cue}</div>
   <div class="bar"><div style="width:${100*frac}%"></div></div><div class="mut" style="margin-top:4px">${Math.max(0,s.dur_s-el).toFixed(1)} s 남음 · 다음: ${nxt?nxt.cue:'끝'}</div>`}
function preview(pv){ const img=$('pv');
  if(!pv.running){img.style.display='none'; $('pvkv').innerHTML=pv.log&&pv.log.length?`<span>상태</span><span>${pv.log.slice(-1)[0]}</span>`:''; return}
  img.style.display='block'; img.src='/preview.jpg?t='+Date.now(); const s=pv.status||{};
  const lamp=v=>`<span class="lamp ${v?'on':'off'}"></span>`;
  $('pvkv').innerHTML=`<span>손 검출</span><span>${lamp(s.detected)}${s.detected?'검출':'없음'} ${s.n_hands!=null?'('+s.n_hands+')':''}</span>
   <span>엄지·검지 끝</span><span>${lamp(s.tips_in_view)}${s.tips_in_view?'화면 안':'벗어남'} · key ${f(s.keys_in_view,0)}/5</span>
   <span>A / B / C</span><span>${f(s.A_px,0)} px · ${f(s.B_ratio)} · ${f(s.C_world,3)} m</span>
   <span>카메라</span><span>${f(s.cam_rate_hz,1)} Hz · 처리 ${f(s.fps,1)} fps</span>
   <span>IMU</span><span>${lamp(s.imu_connected)}${f(s.imu_rate_hz,0)} Hz · age ${f(s.imu_age_ms,0)} ms · |ω| ${f(s.gyro_dps,1)} °/s</span>`}
function eps(list){ $('eps').innerHTML=list.length?list.map(e=>`<div class="ep ${sel===e.path?'sel':''}" data-p="${e.path}"><span>${e.name}<br><span class="mut">${e.wall.slice(0,19)} · ${f(e.duration_s,0)}s ${e.notes?'· '+e.notes:''}</span></span><span class="${e.verdict?(e.verdict.startsWith('SHARE')?'pass':e.verdict.startsWith('INCOMPLETE')?'nm':'fail'):'mut'}">${e.verdict?e.verdict.split(/[:—]/)[0].trim():'미분석'}</span></div>`).join(''):'<div class="mut">아직 녹화 없음</div>';
  document.querySelectorAll('.ep').forEach(d=>d.onclick=()=>{sel=d.dataset.p; $('epPath').value=sel; shownRep=null; loadRep(sel)})}
async function loadRep(ep){ const r=await fetch('/report?ep='+encodeURIComponent(ep)); if(!r.ok){$('rep').innerHTML='<span class="mut">분석 결과 없음 — [분석]을 누르세요.</span>';return}
  const R=await r.json(); shownRep=ep; const v=R.vi||{}, mm=v.masked_motion||{}, um=v.unmasked_motion||{};
  const views=Object.entries(R.views||{}).map(([k,s])=>{const b=(s.signals||{})[s.best_signal]||{};
    const sig=Object.entries(s.signals||{}).map(([n,m])=>`<tr><td>${n}${n===s.best_signal?' ★':''}</td><td>${m.ok?f(m.margin):'–'}</td><td>${m.ok?f(m.dprime,1):'–'}</td><td>${m.ok?f(m.jitter_frac,3):'–'}</td><td>${m.ok?m.chatter:'–'}</td><td>${m.ok?(m.monotonic_half?'예':'아니오'):(m.why||'')}</td></tr>`).join('');
    return `<h3 style="font-size:13px;margin:12px 0 4px">손 · ${k} view <span class="${cls(s.gate_pass)}">${word(s.gate_pass)}</span>${R.hand_best_view===k?' <span class="mut">(판정 사용)</span>':''}</h3>
     <div class="kv"><span>검출 (전체/grip/pinch)</span><span>${f(s.detect_all)} / ${f(s.detect_grip)} / ${f(s.detect_pinch)}</span><span>grip 최장 끊김</span><span>${f(s.miss_run_grip_ms_max,0)} ms</span>
     <span>엄지·검지 끝 화면 안</span><span>${f(s.tips_in_view)}</span><span>처리 시간</span><span>${f(s.ms_per_frame_p50,1)} ms/frame</span></div>
     ${sig?`<table style="margin-top:6px"><tr><th>신호</th><th>margin</th><th>d′</th><th>jitter</th><th>chatter</th><th>HALF 단조</th></tr>${sig}</table>`:''}
     ${b.ok?`<div class="mut" style="margin-top:4px">추천 임계값 (${s.best_signal}): close &lt; ${f(b.close_th,3)} · open &gt; ${f(b.open_th,3)}</div>`:''}
     ${(s.gate_fails||[]).length?`<ul class="fails">${s.gate_fails.map(x=>`<li>${x}</li>`).join('')}</ul>`:''}`}).join('');
  const g=(x)=>x?`r ${f(x.r)} · slope ${f(x.slope)} (n ${x.n})`:'–';
  $('rep').innerHTML=`<div class="verdict ${R.verdict.startsWith('SHARE')?'pass':R.verdict.startsWith('INCOMPLETE')?'nm':'fail'}">${R.verdict}</div>${views}
   <h3 style="font-size:13px;margin:14px 0 4px">VI (마스크 후) <span class="${cls(v.gate_pass)}">${word(v.gate_pass)}</span></h3>
   <div class="kv"><span>남은 환경 / 텍스처 있는 환경</span><span>${f(v.env_ratio)} / ${f(v.env_textured_ratio)}</span>
   <span>inlier p50 / p10 (마스크)</span><span>${f(mm.inliers_p50,0)} / ${f(mm.inliers_p10,0)}  <span class="mut">(마스크 없음 ${f(um.inliers_p50,0)})</span></span>
   <span>grid coverage p50</span><span>${f(mm.grid_cells_p50,0)} / 16</span><span>마스크 없을 때 손 위 inlier 비율</span><span>${f(v.unmasked_inliers_on_hand_frac)}</span>
   <span>회전 vs 자이로 (마스크)</span><span>${g(v.gyro_masked)}</span><span>회전 vs 자이로 (마스크 없음)</span><span>${g(v.gyro_unmasked)}</span><span>IMU lag</span><span>${f(v.imu_lag_ms,0)} ms</span></div>
   ${(v.gate_fails||[]).length?`<ul class="fails">${v.gate_fails.map(x=>`<li>${x}</li>`).join('')}</ul>`:''}
   <div class="imgs">${['overlay.jpg','signals.png','mask_vi.png'].map(n=>`<a href="/file?ep=${encodeURIComponent(ep)}&name=${n}" target="_blank"><img src="/file?ep=${encodeURIComponent(ep)}&name=${n}&t=${Date.now()}" alt="${n}"></a>`).join('')}</div>
   <div class="mut" style="margin-top:6px">기준값은 측정 전 시작값입니다 — 첫 실측 take로 다시 맞추세요.</div>`}
async function tick(){ try{ ST=await (await fetch('/state')).json(); }catch(e){$('meta').textContent='서버 연결 끊김';return}
  $('meta').textContent=`hardware ${ST.hardware}${ST.mock?' · MOCK':''} · ${ST.root}`;
  preview(ST.preview); timeline(ST.protocol,ST.job); recBox(ST.job,ST.protocol); eps(ST.episodes);
  const j=ST.job; const busy=j&&j.running; $('rec').disabled=busy; $('ana').disabled=busy; $('abort').disabled=!busy; $('pvStart').disabled=busy&&j.kind==='record'||ST.preview.running; $('pvStop').disabled=!ST.preview.running;
  if(j){ $('log').textContent=j.log.join('\n'); $('log').scrollTop=1e9;
    if(!j.running && lastJobRc===null){ lastJobRc=j.rc;
      if(j.kind==='record'&&j.episode){sel=j.episode;$('epPath').value=sel}
      if(sel) loadRep(sel)}
    if(j.running) lastJobRc=null }
}
const q=new URLSearchParams(location.search).get('ep'); if(q){sel=q; $('epPath').value=q; loadRep(q)}
setInterval(tick,400); tick();
</script></body></html>"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="G0-0 mount qualification UI")
    ap.add_argument("--port", type=int, default=8720); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--hardware", default="g0_mount_right"); ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--mock", action="store_true", help="mock devices (dry run, 10x faster protocol)")
    ap.add_argument("--open", dest="open_browser", action="store_true")
    a = ap.parse_args(argv)
    a.root.mkdir(parents=True, exist_ok=True)
    app = App(root=a.root.resolve(), hardware=a.hardware, mock=a.mock)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    url = f"http://{a.host}:{a.port}/"; print(f"G0-0 UI on {url}  (hardware {a.hardware}{', MOCK' if a.mock else ''})")
    if a.open_browser: webbrowser.open(url)
    import signal
    # SIGTERM must also stop the children: an orphaned preview keeps the wrist camera open and the next take fails
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        for j in (app.preview, app.job):
            if j: j.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
