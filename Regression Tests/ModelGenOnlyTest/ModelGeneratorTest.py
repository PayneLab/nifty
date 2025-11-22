#!/usr/bin/env python3
"""
Regression test for NIFty: model generation ONLY (using precomputed features).

"""

import subprocess
import sys
import tempfile
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


MODEL_INFO_FILE = "model_information.txt"
MODEL_PKL_FILE = "trained_model_and_model_metadata.pkl"


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


def compare_text(expected_path: Path, actual_path: Path, label: str):
    if not actual_path.exists():
        raise AssertionError(f"{label}: actual file not found: {actual_path}")
    if not expected_path.exists():
        raise AssertionError(f"{label}: expected file not found: {expected_path}")

    exp = expected_path.read_text().strip()
    act = actual_path.read_text().strip()

    if exp != act:
        raise AssertionError(f"{label}: text differs between expected and actual")
    print(f"  ✅ {label}: text matches expected")


def _extract_model(obj):
    if hasattr(obj, "get_params"):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "get_params"):
                return v
    return obj


def compare_models(expected_pkl: Path, actual_pkl: Path, label: str):
    if not actual_pkl.exists():
        raise AssertionError(f"{label}: actual pickle not found: {actual_pkl}")
    if not expected_pkl.exists():
        raise AssertionError(f"{label}: expected pickle not found: {expected_pkl}")

    with expected_pkl.open("rb") as f:
        exp_obj = pickle.load(f)
    with actual_pkl.open("rb") as f:
        act_obj = pickle.load(f)

    exp_model = _extract_model(exp_obj)
    act_model = _extract_model(act_obj)

    # Same type?
    if type(exp_model) is not type(act_model):
        raise AssertionError(f"{label}: model types differ: {type(exp_model)} vs {type(act_model)}")

    # Same hyperparameters?
    if hasattr(exp_model, "get_params") and hasattr(act_model, "get_params"):
        if exp_model.get_params(deep=True) != act_model.get_params(deep=True):
            raise AssertionError(f"{label}: model get_params() differ")

    # Compare common learned attributes, if present
    for attr in ["classes_", "n_features_in_"]:
        exp_val = getattr(exp_model, attr, None)
        act_val = getattr(act_model, attr, None)
        if exp_val is None and act_val is None:
            continue
        if isinstance(exp_val, np.ndarray) or isinstance(act_val, np.ndarray):
            np.testing.assert_array_equal(exp_val, act_val)
        else:
            if exp_val != act_val:
                raise AssertionError(f"{label}: attribute {attr} differs")

    for attr in ["feature_importances_", "coef_", "intercept_"]:
        if hasattr(exp_model, attr) or hasattr(act_model, attr):
            exp_val = getattr(exp_model, attr, None)
            act_val = getattr(act_model, attr, None)
            np.testing.assert_allclose(exp_val, act_val)

    print(f"  ✅ {label}: model object matches expected (type, params, key attributes)")


def main():
    # This file: nifty/Regression Tests/ModelGenOnlyTest/ModelGeneratorTest.py
    test_dir = Path(__file__).resolve().parent
    # Repo root: nifty/
    repo_root = test_dir.parents[1]
    nifty_script = repo_root / "nifty.py"

    original_config = test_dir / "config.toml"
    expected_model_info = test_dir / "expected_model_information.txt"
    expected_model_pkl = test_dir / "expected_trained_model_and_model_metadata.pkl"

    # Basic checks
    if not nifty_script.exists():
        raise FileNotFoundError(f"Could not find nifty.py at {nifty_script}")
    if not original_config.exists():
        raise FileNotFoundError(f"Could not find config.toml at {original_config}")

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

        # 3) Compare outputs

        # model_information.txt
        actual_model_info = temp_dir / MODEL_INFO_FILE
        compare_text(expected_model_info, actual_model_info, "Model information")

        # trained_model_and_model_metadata.pkl
        actual_model_pkl = temp_dir / MODEL_PKL_FILE
        compare_models(expected_model_pkl, actual_model_pkl, "Trained model")

        print("\n✅ ModelGeneratorTest regression test PASSED — all outputs match expected.")


if __name__ == "__main__":
    main()
