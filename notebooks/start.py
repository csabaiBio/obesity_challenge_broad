import sys
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
print(f"Root set to: {ROOT}")

