#!/usr/bin/env python3
"""[2026-09-27] Config parity from the FULL config LeRobot prints at startup (`ot_train.py:212 {...}`), because train_config.json is
only written at the first checkpoint. Usage: cfg_parity.py <reference log> <new log> <comma-separated allowed keys> [label]
Exit 0 iff every flattened key is equal except the allowed ones and the new run has resume == False."""
import ast, json, os, re, sys
def cfg_from_log(path):
    t = open(path, errors="replace").read().replace("\r", "\n"); i = t.index("ot_train.py:212 {"); s = t[t.index("{", i):]; depth = 0
    for j, ch in enumerate(s):
        depth += ch == "{"; depth -= ch == "}"
        if depth == 0: s = s[:j + 1]; break
    s = re.sub(r"<(\w+)\.(\w+): '([^']*)'>", r"'\3'", s); s = re.sub(r"<(\w+)\.(\w+): (\d+)>", r"\3", s)
    return ast.literal_eval(s)
def flat(d, p=""):
    o = {}
    for k, v in d.items():
        if isinstance(v, dict): o.update(flat(v, p + k + "."))
        else: o[p + k] = v
    return o
nb = cfg_from_log(sys.argv[2])
if os.environ.get("CFG_DUMP"): json.dump(nb, open(os.environ["CFG_DUMP"], "w"), indent=1)
a, b = flat(cfg_from_log(sys.argv[1])), flat(nb); allow = set(filter(None, sys.argv[3].split(","))); label = sys.argv[4] if len(sys.argv) > 4 else ""
diff = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
bad = {k: v for k, v in diff.items() if k not in allow}
print(f"[parity {label}] {len(a)} vs {len(b)} keys | allowed diffs: " + ", ".join(f"{k}: {str(v[0])[:40]} -> {str(v[1])[:40]}" for k, v in sorted(diff.items()) if k in allow))
print(f"[parity {label}] resume={b.get('resume')} steps={b.get('steps')} decay={b.get('policy.scheduler_decay_steps')} seed={b.get('seed')} episodes={len(b.get('dataset.episodes') or [])}")
ok = not bad and b.get("resume") is False
if bad: print(f"[parity {label}] DISALLOWED: {bad}")
print(f"[parity {label}] CONFIG PARITY {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
