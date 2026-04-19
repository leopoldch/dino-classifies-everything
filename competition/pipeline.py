#!/usr/bin/env python
"""Train un modele (3 seeds) puis genere la submission ensembliste."""
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

SPECIALIST_IMAGE_SIZE = 384


def run(cmd):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS.keys())
    parser.add_argument("--skip-submit", action="store_true", help="Ne pas generer la submission")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(exist_ok=True)

    script = MODELS[args.model]
    print(f"{script}")
    run([sys.executable, str(ROOT / script)])

    # deplacer les checkpoints finaux vers weights/
    for pt in ROOT.glob(f"{args.model}-s*-final.pt"):
        dest = WEIGHTS_DIR / pt.name
        shutil.move(str(pt), str(dest))
        print(f"  {pt.name} -> weights/")

    # nettoyer les checkpoints intermediaires
    for pt in list(ROOT.glob(f"{args.model}-s*-head.pt")) + list(ROOT.glob(f"{args.model}-s*-partial.pt")):
        pt.unlink()

    if args.skip_submit:
        return

    # trouver tous les checkpoints finaux dans weights/
    finals = sorted(WEIGHTS_DIR.glob("*-s*-final.pt"))
    specialist = next(WEIGHTS_DIR.glob("montreal-specialist*final.pt"), None)

    if not finals:
        print("Aucun checkpoint final trouve")
        sys.exit(1)

    print(f"\nEnsemble: {len(finals)} checkpoints")
    for f in finals:
        print(f"  {f.name}")

    cmd = [sys.executable, str(ROOT / "make_submission.py")]
    cmd += [str(f) for f in finals]
    if specialist:
        print(f"Specialiste: {specialist.name}")
        cmd += ["--montreal-specialist", str(specialist)]
        # le specialiste a ete entraine a 384px, pas 392
        cmd += ["--montreal-specialist-image-size", str(SPECIALIST_IMAGE_SIZE)]
    cmd += ["--output", "submission.csv"]

    run(cmd)
    print("\nsubmission.csv genere")


if __name__ == "__main__":
    main()
