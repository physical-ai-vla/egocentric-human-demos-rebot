"""Operator dashboard (2026-10-14, spec 35-36). One page: RGB-D + both wrist views, per-hand tracking / IMU / pinch, deadman,
clutch, gripper L/R, robot mode, target vs executed TCP, latency, and THE reason the robot is holding -- in words, at the top.

Keys (page focused): SPACE held = deadman (heartbeat every 50 ms; release, blur or a lost tab = released after 150 ms),
C = clutch toggle, ESC = E-STOP (latched; RESET needs every source clear + re-INIT).
Served by Flask from the control process (live_teleop.py); images are the JPEGs the sensor processes write to /tmp."""
from __future__ import annotations
import json
from pathlib import Path

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Visual Teleop</title>
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--ok:#2e7d32;--warn:#b58900;--bad:#c62828;--card:#1d1d1d}
body{background:var(--bg);color:var(--fg);font:14px -apple-system,Helvetica,sans-serif;margin:12px}
#banner{font-size:26px;font-weight:700;padding:10px 14px;border-radius:8px;margin-bottom:10px}
.ok{background:var(--ok)}.warn{background:var(--warn)}.bad{background:var(--bad)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:10px}
.card{background:var(--card);border-radius:8px;padding:10px}.card h3{margin:0 0 6px 0;font-size:15px}
img{width:100%;border-radius:6px;background:#000;min-height:120px}
table{width:100%;border-collapse:collapse}td{padding:2px 4px;border-bottom:1px solid #2a2a2a;font-family:ui-monospace,monospace;font-size:12px}
button{font-size:15px;padding:8px 14px;margin:3px;border-radius:6px;border:0;cursor:pointer}
.big{font-size:20px;padding:12px 20px}#estop{background:var(--bad);color:#fff}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;margin-right:4px}
</style></head><body>
<div id="banner" class="bad">connecting…</div>
<div>
 <button id="estop" class="big" onclick="post('/estop')">비상정지 (Esc)</button>
 <button onclick="post('/estop_reset')">비상정지 해제</button>
 <button onclick="post('/init')">시작 맞추기 (INIT)</button>
 <button onclick="post('/clutch',{on:!S.clutch})">클러치 (C)</button>
 <button onclick="post('/rec_start')">녹화 시작</button><button onclick="post('/rec_stop')">녹화 끝</button>
 <span id="dm" class="pill bad">정지 중 — 버튼이나 스페이스바를 누르고 있으세요</span><span id="live" class="pill"></span>
</div>
<button id="hold" style="width:100%;font-size:28px;padding:22px;margin:8px 0;background:#2e7d32;color:#fff;border-radius:10px;user-select:none">누르고 있는 동안만 로봇 동작 (손 떼면 즉시 정지) — 스페이스바도 됨</button>
<div class="grid">
 <div class="card"><h3>fixed RGB-D</h3><img id="i_rgbd"></div>
 <div class="card"><h3>left wrist</h3><img id="i_left"></div>
 <div class="card"><h3>right wrist</h3><img id="i_right"></div>
</div>
<div class="grid" style="margin-top:10px">
 <div class="card"><h3>left hand</h3><table id="t_left"></table></div>
 <div class="card"><h3>right hand</h3><table id="t_right"></table></div>
 <div class="card"><h3>robot</h3><table id="t_robot"></table></div>
</div>
<script>
let S={}, held=false, hb=null;
function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})}).then(r=>r.json()).then(j=>{if(j.error)alert(j.error)}).catch(e=>alert(e))}
let relT=null;
function press(){if(relT){clearTimeout(relT);relT=null} if(held)return;held=true;fetch('/hb',{method:'POST'});hb=setInterval(()=>fetch('/hb',{method:'POST'}),50)}
function release(){held=false;if(hb)clearInterval(hb);hb=null;fetch('/release',{method:'POST'})}
// [10-15] with the Korean IME active the SPACE keydown never arrived (only keyup/release did): accept every space signature,
// and offer a hold-to-run button that does not depend on the keyboard at all.
const isSpace=e=>e.code==='Space'||e.key===' '||e.keyCode===32;
document.addEventListener('keydown',e=>{if(isSpace(e)){e.preventDefault();press()} if(e.code==='Escape'||e.key==='Escape')post('/estop'); if((e.code==='KeyC'||e.key==='c')&&!e.repeat)post('/clutch',{on:!S.clutch})},true);
// [10-16] robot8: 143 deadman releases, 136 shorter than 0.6 s (p50 one tick) while SPACE was held -- key repeat / IME sends
// keyup+keydown pairs, every instant /release re-anchored the arm (FOLLOW runs p50 0.36 s). A SPACE keyup now releases after 150 ms
// unless SPACE comes back; blur / hidden tab / pointer release stay instant.
document.addEventListener('keyup',e=>{if(isSpace(e)){if(relT)clearTimeout(relT);relT=setTimeout(()=>{relT=null;release()},150)}},true);
const hb_btn=document.getElementById('hold');
hb_btn.addEventListener('pointerdown',e=>{e.preventDefault();hb_btn.setPointerCapture(e.pointerId);press()});
['pointerup','pointercancel','lostpointercapture'].forEach(t=>hb_btn.addEventListener(t,release));
window.addEventListener('blur',release);document.addEventListener('visibilitychange',()=>{if(document.hidden)release()});
const f=(x,n=3)=>x==null?'—':(typeof x==='number'?x.toFixed(n):Array.isArray(x)?x.map(v=>typeof v==='number'?v.toFixed(n):v).join(', '):String(x));
function rows(id,o){document.getElementById(id).innerHTML=Object.entries(o).map(([k,v])=>`<tr><td>${k}</td><td>${f(v)}</td></tr>`).join('')}
async function tick(){
 try{S=await (await fetch('/status')).json()}catch(e){document.getElementById('banner').textContent='control process not reachable';document.getElementById('banner').className='bad';return}
 const b=document.getElementById('banner');b.textContent=S.headline;b.className=S.severity;
 const d=document.getElementById('dm');d.textContent=S.deadman?'동작 허용 중':'정지 중 — 버튼이나 스페이스바를 누르고 있으세요';d.className='pill '+(S.deadman?'ok':'bad');
 const l=document.getElementById('live');l.textContent=S.live?'실제 로봇':'미리보기 (로봇 명령 없음)';l.className='pill '+(S.live?'bad':'ok');
 for(const s of ['left','right'])rows('t_'+s,S.hands[s]||{});
 rows('t_robot',S.robot||{});
 const t=Date.now();for(const n of ['rgbd','left','right'])document.getElementById('i_'+n).src='/img/'+n+'?t='+t;
}
setInterval(tick,200);tick();
</script></body></html>"""


def make_app(get_status, actions: dict, img_paths: dict):
    from flask import Flask, jsonify, request, send_file, Response
    app = Flask("visual_teleop")

    @app.get("/")
    def index(): return Response(PAGE, mimetype="text/html")

    @app.get("/status")
    def status(): return Response(json.dumps(get_status(), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), mimetype="application/json")

    @app.get("/view/<name>")
    def view(name):
        """Full-window live view of one camera (10-15: for aiming the fixed RGB-D)."""
        return Response(f"""<!doctype html><html><head><meta charset=utf-8><title>{name} live</title>
<style>body{{margin:0;background:#000;color:#ff0;font:16px sans-serif}}img{{width:100vw;height:100vh;object-fit:contain}}
#t{{position:fixed;top:6px;left:8px}}</style></head><body><div id=t></div><img id=i>
<script>let n=0;setInterval(()=>{{i.src='/img/{name}?t='+Date.now();t.textContent='{name} live '+(++n)}},150)</script></body></html>""",
                        mimetype="text/html")

    @app.get("/img/<name>")
    def img(name):
        p = Path(img_paths.get(name, ""))
        return send_file(p, mimetype="image/jpeg", max_age=0) if p.is_file() else ("", 404)

    for path, fn in actions.items():
        def handler(fn=fn):
            try: return jsonify(fn(request.get_json(silent=True) or {}) or {"ok": True})
            except Exception as e: return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 400
        app.add_url_rule(path, path.strip("/").replace("/", "_") or "root", handler, methods=["POST"])
    return app
