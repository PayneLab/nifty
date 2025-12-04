"""
Regression test for NIFty: APPLY MODEL ONLY.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


PRED_CLASSES_FILE = "predicted_classes.tsv"


def make_temp_config(original_config: Path, temp_dir: Path) -> Path:
    """
    Read the original config.toml and write a copy in temp_dir
    with output_dir set to temp_dir.
    """
    text = original_config.read_text()

    new_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("output_dir"):
            new_lines.append(f'output_dir = "{temp_dir.as_posix()}"')
        else:
            new_lines.append(line)

    temp_config = temp_dir / "config.toml"
    temp_config.write_text("\n".join(new_lines) + "\n")
    return temp_config


def run_nifty(nifty_script: Path, config_path: Path, repo_root: Path) -> subprocess.CompletedProcess:
    """
    Run `python nifty.py -c <config>` in the repo_root directory.
    """
    cmd = [sys.executable, str(nifty_script), "-c", str(config_path)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    print("\n=== STDOUT ===")
    print(result.stdout or "(no stdout)")
    print("\n=== STDERR ===")
    print(result.stderr or "(no stderr)")

    return result


def compare_predictions(expected_path: Path, actual_path: Path):
    if not actual_path.exists():
        raise AssertionError(f"Predictions: actual file not found: {actual_path}")
    if not expected_path.exists():
        raise AssertionError(f"Predictions: expected file not found: {expected_path}")

    exp = pd.read_csv(expected_path, sep="\t")
    act = pd.read_csv(actual_path, sep="\t")

    # check_like=True ignores column order; index is ignored by default
    pd.testing.assert_frame_equal(exp, act, check_like=True)
    print("  ✅ Predictions: TSV matches expected")


def main():
    test_dir = Path(__file__).resolve().parent

    repo_root = test_dir.parents[1]
    nifty_script = repo_root / "nifty.py"

    original_config = test_dir / "config.toml"
    expected_pred_classes = test_dir / "expected_predicted_classes.tsv"

    # Sanity checks
    if not nifty_script.exists():
        raise FileNotFoundError(f"Could not find nifty.py at {nifty_script}")
    if not original_config.exists():
        raise FileNotFoundError(f"Could not find config.toml at {original_config}")
    if not expected_pred_classes.exists():
        raise FileNotFoundError(f"Missing expected_predicted_classes.tsv at {expected_pred_classes}")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        temp_dir = Path(tmpdir_str)
        print(f"Using temporary output directory: {temp_dir}")

        # 1) Create temp config that writes outputs into temp_dir
        temp_config = make_temp_config(original_config, temp_dir)

        # 2) Run NIFty with the temp config
        result = run_nifty(nifty_script, temp_config, repo_root)

        if result.returncode != 0:
            print("\n❌ NIFty exited with non-zero status:", result.returncode)
            sys.exit(1)

        # 3) Compare predicted classes output
        actual_pred_classes = temp_dir / PRED_CLASSES_FILE
        try:
            compare_predictions(expected_pred_classes, actual_pred_classes)
        except AssertionError as e:
            print("\n❌ ApplyClassifierOnlyTest regression FAILED:")
            print(e)
            sys.exit(1)

        print("\n✅ ApplyClassifierOnlyTest regression PASSED — predictions match expected.")


if __name__ == "__main__":
    main()
