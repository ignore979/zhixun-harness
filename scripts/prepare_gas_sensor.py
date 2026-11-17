"""Command line converter; source and conversion evidence remain separate from received data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zhixun.prepare_gas import prepare_gas

parser = argparse.ArgumentParser()
parser.add_argument("--archive", type=Path, required=True)
args = parser.parse_args()
print(json.dumps(prepare_gas(args.archive), ensure_ascii=False, indent=2))
