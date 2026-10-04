#!/usr/bin/env python3
"""usage: convert_episode.py <raw_episode_dir> <out_episode_dir>   (one episode; same code path as convert_dataset)"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from ego_cart20.convert import convert_episode
from ego_cart20.io.processed_writer import write_episode
from ego_cart20.io.raw_episode_loader import load_raw_episode
ep = convert_episode(load_raw_episode(sys.argv[1])); write_episode(sys.argv[2], ep)
print(json.dumps({k: ep["metadata"][k] for k in ("episode_id", "stack_order", "frames_rows", "rows_valid", "train_rows", "duration_s")}))
