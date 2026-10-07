"""[2026-09-29] lerobot_train with the REL16-v3 aux plugin installed first (rel16_aux.py). Same CLI as lerobot_train.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rel16_aux
rel16_aux.install()
from lerobot.scripts.lerobot_train import main
main()
