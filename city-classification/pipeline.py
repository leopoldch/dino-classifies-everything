#!/usr/bin/env python
"""Train a model (3 seeds) then generate the ensemble submission."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"

MODELS = {
    "dinov3-evolved": "models/dinov3-evolved.py",
    "dinov3-gem": "models/dinov3-gem.py",
    "dinov2-evolved": "models/dinov2-evolved.py",
}

SPECIALIST_IMAGE_SIZE = 392


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS.keys())
    parser.add_argument("--skip-submit", action="store_true", help="Skip submission generation")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)

    script = MODELS[args.model]
    print(f"{script}")
    run([sys.executable, str(ROOT / script)])

    # move final checkpoints to weights/
    for pt in ROOT.glob(f"{args.model}-s*-final.pt"):
        dest = WEIGHTS_DIR / pt.name
        shutil.move(str(pt), str(dest))
        print(f"  {pt.name} -> weights/")

    # clean up intermediate checkpoints
    for pt in list(ROOT.glob(f"{args.model}-s*-head.pt")) + list(ROOT.glob(f"{args.model}-s*-partial.pt")):
        pt.unlink()

    if args.skip_submit:
        return

    # find all final checkpoints in weights/
    finals = sorted(WEIGHTS_DIR.glob("*-s*-final.pt"))
    specialist = next(WEIGHTS_DIR.glob("montreal-specialist*final.pt"), None)

    if not finals:
        print("No final checkpoints found")
        sys.exit(1)

    print(f"\nEnsemble: {len(finals)} checkpoints")
    for f in finals:
        print(f"  {f.name}")

    cmd = [sys.executable, str(ROOT / "make_submission.py")]
    cmd += [str(f) for f in finals]
    if specialist:
        print(f"Specialist: {specialist.name}")
        cmd += ["--montreal-specialist", str(specialist)]
        cmd += ["--montreal-specialist-image-size", str(SPECIALIST_IMAGE_SIZE)]
    cmd += ["--output", "submission.csv"]

    run(cmd)
    print("\nsubmission.csv generated")


if __name__ == "__main__":
    main()
