#!/usr/bin/env python3
"""validate_dataset.py with the Soft-Fold chunk-32 contract (softfold_h32 imported first).  usage: <processed_root> [report.json]"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import ego_cart20.softfold_h32  # noqa: F401
from ego_cart20.validation.integrity import validate_dataset
rep = validate_dataset(sys.argv[1]); out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(sys.argv[1]) / "integrity_report.json")
json.dump(rep, open(out, "w"), indent=1); print(json.dumps(rep, indent=1)); print("VALIDATION", "PASS" if rep["pass"] else f"FAIL {rep['fails']}")
sys.exit(0 if rep["pass"] else 1)
