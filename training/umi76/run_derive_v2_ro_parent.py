"""[2026-09-30] Run derive_v2.py UNCHANGED on a read-only (frozen) parent: its meta copy uses shutil.copy2, which keeps the parent's
read-only mode, so the later rewrite of meta/stats.json fails (EPERM). derive_v3d / derive_relcart20 already do copy2 + chmod 0o644;
this wrapper applies exactly that to derive_v2. Only the COPIED files (separate inodes) are chmod'ed; hard-linked videos are untouched."""
import os, runpy, shutil, sys
_copy2 = shutil.copy2
def copy2_rw(src, dst, *a, **k):
    r = _copy2(src, dst, *a, **k); os.chmod(r, 0o644); return r
shutil.copy2 = copy2_rw
sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "derive_v2.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
