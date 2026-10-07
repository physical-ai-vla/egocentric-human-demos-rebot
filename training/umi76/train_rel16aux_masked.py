"""[2026-09-30] lerobot_train with rel16_aux_masked installed first (ego cart-only pretrain; also valid for real v4 FT = bit-identical loss). Same CLI as lerobot_train.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_aux_masked
rel16_aux_masked.install()
from lerobot.scripts.lerobot_train import main
main()
