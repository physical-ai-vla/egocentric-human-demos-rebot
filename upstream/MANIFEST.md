# upstream/ MANIFEST

Assembled 2026-10-07. Only read-only git commands were run on the source repos (`rev-parse`, `diff`, `ls-files`,
`status`, `log`, `archive`).

| source | dest | notes |
|---|---|---|
| `git -C ~/handumi-sw diff -- . ':(exclude)uv.lock'` | `handumi-sw-rebot-ego/tracked.patch` | 8 files, +111/−8 |
| same, `--stat` | `handumi-sw-rebot-ego/tracked.diffstat.txt` | |
| `git -C ~/handumi-sw ls-files --others --exclude-standard` (minus `.DS_Store`) | `handumi-sw-rebot-ego/new_files.list` | 33 paths |
| those 33 files | `handumi-sw-rebot-ego/new_files/<same path>` | Paths preserved. The largest is `src/handumi/tracking/apriltag.py` at 41 KB. |
| `~/handumi-sw/configs/rig.yaml` (gitignored upstream) | `handumi-sw-rebot-ego/new_files/configs/rig.yaml` | The local rig config the fork needs. It contains a LAN IP; it has no secrets. |
| `~/handumi-sw/LICENSE` | `LICENSE-handumi-apache-2.0.txt` | Verbatim |
| `~/handumi-hw/LICENSE` | `LICENSE-handumi-hw-apache-2.0.txt` | Verbatim. It differs from the sw LICENSE in its appendix and notices. |
| (new) | `README.md`, `NOTICE.md`, `MANIFEST.md` | |

## Base commits

- handumi-sw: `33cc437229e43ed4bb92b25d4465248f0edbbcd9`. The local branch is `rebot-ego`, at the same commit as
  `origin/main`. All changes are uncommitted.
- handumi-hw: `e58de33b9a88bd208e1a40e52ea575eaf68963db` (`main`). It has no local changes.

## Excluded

| item | reason |
|---|---|
| `uv.lock` diff | A regenerated lockfile (+45/−45 marker re-resolve) |
| `.DS_Store`, `.venv/`, `.pytest_cache/`, `__pycache__/` | Cruft and ignored build state |
| Files larger than 5 MB | None found, so nothing was skipped |
| `~/handumi-hw/result.json` | A Bambu CLI slicer result stub, not a design change |

## Verification

A clean `git archive 33cc437`, with `tracked.patch` applied and `new_files/` copied over, gives a tree identical to
`~/handumi-sw`. `diff -r` was clean, excluding `.venv`, `.git`, caches and `uv.lock`.
