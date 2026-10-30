import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Optional


def _load_json(path: Optional[str]) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _append_opt(cmd: list, flag: str, value) -> None:
    if value is None:
        return
    cmd.extend([flag, str(value)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", dest="out", required=True)
    parser.add_argument("--repo", dest="repo", required=False, default="")
    parser.add_argument("--config", dest="config", required=False, default=None)
    parser.add_argument("--seed", dest="seed", type=int, required=False, default=42)
    args = parser.parse_args()

    self_dir = os.path.dirname(os.path.abspath(__file__))
    bridge = os.path.join(self_dir, "scprotein_adapter.py")
    if not os.path.exists(bridge):
        sys.stderr.write(f"scprotein bridge script not found: {bridge}\n")
        return 2

    cfg = _load_json(args.config)
    cmd = [sys.executable, bridge, "--input", os.path.abspath(args.inp), "--out", os.path.abspath(args.out)]
    _append_opt(cmd, "--seed", args.seed)
    _append_opt(cmd, "--threshold", cfg.get("threshold"))
    _append_opt(cmd, "--num-epochs", cfg.get("num_epochs"))
    _append_opt(cmd, "--num-hidden", cfg.get("num_hidden"))
    _append_opt(cmd, "--num-proj-hidden", cfg.get("num_proj_hidden"))
    _append_opt(cmd, "--num-layers", cfg.get("num_layers"))
    _append_opt(cmd, "--num-protos", cfg.get("num_protos"))
    _append_opt(cmd, "--alpha", cfg.get("alpha"))
    _append_opt(cmd, "--tau", cfg.get("tau"))
    _append_opt(cmd, "--learning-rate", cfg.get("learning_rate"))
    _append_opt(cmd, "--weight-decay", cfg.get("weight_decay"))
    _append_opt(cmd, "--device", cfg.get("device"))

    try:
        proc = subprocess.run(cmd, check=False)
        return int(proc.returncode)
    except Exception as exc:
        sys.stderr.write(f"failed to run scprotein adapter: {exc}\n")
        return 3


if __name__ == "__main__":
    sys.exit(main())
