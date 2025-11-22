#!/usr/bin/env python3
"""
Regression test for NIFty feature selection.

"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


OUTPUT_FILENAME = "selected_features.tsv"


def make_temp_config(original_config: Path, temp_dir: Path) -> Path:
    """
    Read the original config.toml and write a copy to temp_dir
    with output_dir set to temp_dir.
    """
    text = original_config.read_text()

    new_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("output_dir"):
            # Override output_dir to be an absolute path to temp_dir
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
    print("=== STDERR ===")
    print(result.stderr or "(no stderr)")

    return result


def compare_outputs(expected_path: Path, actual_path: Path) -> None:
    if not actual_path.exists():
        raise FileNotFoundError(f"Actual output file not found: {actual_path}")

    if not expected_path.exists():
        raise FileNotFoundError(f"Expected output file not found: {expected_path}")

    expected_df = pd.read_csv(expected_path, sep="\t")
    actual_df = pd.read_csv(actual_path, sep="\t")

    pd.testing.assert_frame_equal(expected_df, actual_df, check_like=True)


def main():
    # This file: nifty/Regression Tests/FeatureSelectionTest/FeatureSelectionTest.py
    test_dir = Path(__file__).resolve().parent
    # Repo root: nifty/
    repo_root = test_dir.parents[1]
    nifty_script = repo_root / "nifty.py"

    original_config = test_dir / "config.toml"
    expected_output = test_dir / "expected_selected_features.tsv"

    if not nifty_script.exists():
        raise FileNotFoundError(f"Could not find nifty.py at {nifty_script}")

    if not original_config.exists():
        raise FileNotFoundError(f"Could not find config.toml at {original_config}")

    if not expected_output.exists():
        raise FileNotFoundError(f"Could not find expected_selected_features.tsv at {expected_output}")

    with tempfile.TemporaryDirectory() as tmpdir_str:
        temp_dir = Path(tmpdir_str)
        print(f"Using temporary output directory: {temp_dir}")

        # 1) Create a temp config that writes into temp_dir
        temp_config = make_temp_config(original_config, temp_dir)

        # 2) Run NIFty with the temp config
        result = run_nifty(nifty_script, temp_config, repo_root)

        if result.returncode != 0:
            print("\n❌ NIFty exited with non-zero status:", result.returncode)
            sys.exit(1)

        # 3) Compare actual vs expected outputs
        actual_output = temp_dir / OUTPUT_FILENAME

        try:
            compare_outputs(expected_output, actual_output)
        except AssertionError as e:
            print("\n❌ Regression test FAILED: output differs from expected.")
            print(str(e))
            sys.exit(1)

        print("\n✅ Regression test PASSED: output matches expected.")


if __name__ == "__main__":
    main()
