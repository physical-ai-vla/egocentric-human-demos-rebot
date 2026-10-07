#!/usr/bin/env python3
"""Drop the reBot-feasibility result where the collector can see it, so the quota counts usable data.

Canonical criterion (2026-09-22): **per-arm AND** -- each arm's own 17-step p95 must pass, and both must.
Pooled p95 over 34 values is NOT used: one arm's good steps dilute the other's bad ones, which made the
pooled figure (37.5%) look better than the honest one (32.7%).

Measured on 2,241 bimanual chunks: left 51.7%, right 67.7%, both 32.7% ~= 0.517 x 0.677. The two arms are
kinematically independent (coupled and independent IK gave identical results), so bimanual feasibility is
the PRODUCT of the two. That is why left/right are tracked separately here -- raising the weaker arm moves
the bimanual number quadratically, and a per-order shortfall is usually one arm, not both.

This writes `<session>/feasible_counters.json`. `EpisodeManager` reads it every time it is consulted and
falls back to raw counts when absent, so the audit never blocks recording.

No inclusion threshold is baked in beyond `--min-chunks`. `feasible_chunks`, `total_bimanual_chunks` and
`feasible_fraction` are all stored per episode so a later criterion -- say `min_chunks AND fraction >= X`
-- can be applied without re-running the audit. Episode length varies, so a flat chunk count is strict on
short episodes and loose on long ones; keep both numbers.

    python write_feasible_counters.py --session <dir> --criterion bimanual_criterion.json [--min-chunks 5]
"""
import argparse, collections, json, pathlib

ACCEPT = {"PASS", "SOFT_PASS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--criterion", required=True, help="bimanual_criterion.json (per-arm labels)")
    ap.add_argument("--min-chunks", type=int, default=5,
                    help="feasible bimanual chunks an episode needs to count toward the quota")
    a = ap.parse_args()

    sess = pathlib.Path(a.session)
    stamp = sess.name.replace("Hpilot_", "")
    crit = json.loads(pathlib.Path(a.criterion).read_text())

    tot = collections.Counter()
    good = {k: collections.Counter() for k in ("left", "right", "both")}
    for c in crit["chunks"]:
        if not c["src"].startswith(stamp + "_"):
            continue
        tot[c["src"]] += 1
        for k in ("left", "right", "both"):
            if c[k] in ACCEPT:
                good[k][c["src"]] += 1

    order_of = {}
    for ep in sorted(sess.glob("episode_*")):
        m = ep / "episode_meta.json"
        if m.is_file():
            j = json.loads(m.read_text())
            if j.get("order"):
                order_of[f"{stamp}_{ep.name.replace('episode_', '')}"] = j["order"]

    orders = sorted(set(order_of.values()))
    z = lambda: {o: 0 for o in orders}
    feasible, feas_l, feas_r = z(), z(), z()
    chunks_b, chunks_l, chunks_r, chunks_tot = z(), z(), z(), z()
    episodes = []
    for ep, n_tot in tot.items():
        o = order_of.get(ep)
        if not o:
            continue
        gb, gl, gr = good["both"][ep], good["left"][ep], good["right"][ep]
        chunks_tot[o] += n_tot; chunks_b[o] += gb; chunks_l[o] += gl; chunks_r[o] += gr
        if gb >= a.min_chunks:
            feasible[o] += 1
        if gl >= a.min_chunks:
            feas_l[o] += 1
        if gr >= a.min_chunks:
            feas_r[o] += 1
        episodes.append(dict(episode=ep, order=o, total_bimanual_chunks=n_tot,
                             feasible_chunks=gb, feasible_chunks_left=gl, feasible_chunks_right=gr,
                             feasible_fraction=round(gb / max(n_tot, 1), 4),
                             feasible_fraction_left=round(gl / max(n_tot, 1), 4),
                             feasible_fraction_right=round(gr / max(n_tot, 1), 4)))

    raw = collections.Counter(order_of.values())
    (sess / "feasible_counters.json").write_text(json.dumps(dict(
        schema="handumi_feasible_counters/v2",
        criterion="per-arm AND (left p95 pass AND right p95 pass); pooled p95 deliberately not used",
        source=str(pathlib.Path(a.criterion).resolve()),
        min_chunks_per_episode=a.min_chunks,
        feasible=feasible, feasible_left=feas_l, feasible_right=feas_r,
        feasible_chunks=chunks_b, feasible_chunks_left=chunks_l, feasible_chunks_right=chunks_r,
        total_bimanual_chunks=chunks_tot, raw_episodes=dict(raw),
        episodes=sorted(episodes, key=lambda e: e["episode"]),
        note="FAIL episodes stay on disk; a solver or anchor change may reclassify them. Thresholds other "
             "than min_chunks are intentionally not baked in -- feasible_fraction is stored so a later "
             "rule can be applied without re-auditing."), indent=1))

    print(f"-> {sess / 'feasible_counters.json'}")
    print(f"   {'order':<7} {'raw':>4} {'bimanual ep':>12} {'L feas':>7} {'R feas':>7} {'BOTH feas':>10} "
          f"{'chunks L/R/both':>20}")
    for o in orders:
        nb = sum(1 for e in episodes if e["order"] == o)
        print(f"   {o:<7} {raw[o]:>4} {nb:>12} {feas_l[o]:>7} {feas_r[o]:>7} {feasible[o]:>10} "
              f"{f'{chunks_l[o]}/{chunks_r[o]}/{chunks_b[o]}':>20}")
    print(f"   {'TOTAL':<7} {sum(raw.values()):>4} {len(episodes):>12} {sum(feas_l.values()):>7} "
          f"{sum(feas_r.values()):>7} {sum(feasible.values()):>10} "
          f"{f'{sum(chunks_l.values())}/{sum(chunks_r.values())}/{sum(chunks_b.values())}':>20}")
    cl, cr, cb, ct = (sum(x.values()) for x in (chunks_l, chunks_r, chunks_b, chunks_tot))
    if ct:
        print(f"\n   chunk feasibility  left {cl/ct*100:.1f}%  right {cr/ct*100:.1f}%  both {cb/ct*100:.1f}%"
              f"   (곱 {cl/ct*cr/ct*100:.1f}% — 두 팔은 독립이므로 약한 쪽을 올리는 게 제곱으로 이득)")
    weak = "left" if cl < cr else "right"
    print(f"   개선 우선순위: {weak} (현재 {min(cl,cr)/max(ct,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
