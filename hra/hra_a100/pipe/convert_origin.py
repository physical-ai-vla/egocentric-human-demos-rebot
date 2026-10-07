"""[2026-10-07] right_only convert with anchor = IDENTITY (the raw poses are already in the origin frame F, make_origin_frame.py), so the
RELCART20 state = absolute pose in F. Actions are unchanged (relative to the current pose). usage: convert_origin.py <raw> <out> [val_frac=0.1]   ([2026-10-07 user] 0 = all episodes in train)"""
import inspect, pathlib, sys
from ego_cart20 import right_only as RO
src = inspect.getsource(RO.convert_right_only)
assert "anchor = Tr[0].copy()" in src
exec(compile(src.replace("anchor = Tr[0].copy()", "anchor = np.eye(4)  # [origin variant] F frame"), RO.__file__, "exec"), RO.__dict__)
print(RO.convert_tree(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), val_frac=float(sys.argv[3]) if len(sys.argv) > 3 else 0.1))
