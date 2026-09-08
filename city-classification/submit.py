import subprocess
import sys
from pathlib import Path
from config import Config

config = Config()
COMPETITION = config.COMPETITION


def submit(csv_path: Path, message: str = "submission"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"file not found : {csv_path}")
        sys.exit(1)

    try:
        subprocess.run(
            ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", str(csv_path), "-m", message],
            check=True,
        )
        print("submission done")
    except subprocess.CalledProcessError:
        print("error submitting")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to csv submision file")
    parser.add_argument("-m", "--message", default="submission", help="Submission message")
    args = parser.parse_args()

    submit(Path(args.csv), args.message)
