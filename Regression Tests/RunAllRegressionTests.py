#!/usr/bin/env python3
"""
Master regression runner for NIFty.

"""

import subprocess
import sys
from pathlib import Path


def run_test(name: str, script_path: Path, cwd: Path):
    print("\n" + "=" * 70)
    print(f" RUNNING {name}")
    print("=" * 70)
    print(f"Script: {script_path}\n")

    if not script_path.exists():
        print(f"❌ SKIPPING {name}: script not found at {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    # Echo child output
    print("---- STDOUT ----")
    print(result.stdout or "(no stdout)")
    print("---- STDERR ----")
    print(result.stderr or "(no stderr)")

    if result.returncode == 0:
        print(f"\n✅ {name} PASSED")
        return True
    else:
        print(f"\n❌ {name} FAILED with exit code {result.returncode}")
        return False


def main():
    base_dir = Path(__file__).resolve().parent  # .../nifty/Regression Tests
    # We'll run each script from base_dir so their own path logic still works
    cwd = base_dir

    TEST_SCRIPTS = [
        ("Test 1: FS only",
         base_dir / "FeatureSelectionTest" / "FeatureSelectionTest.py"),
        ("Test 2: FS + Train (ModelGeneratorTest)",
         base_dir / "ModelGeneratorTest" / "ModelGeneratorTest.py"),
        ("Test 3: FS + Train + Apply (ApplyClassifierTest)",
         base_dir / "ApplyClassifierTest" / "ApplyClassifierTest.py"),
        ("Test 4: Train only, given features (ModelGenOnlyTest)",
         base_dir / "ModelGenOnlyTest" / "ModelGeneratorTest.py"),
        ("Test 5: Apply only, given features + model (ApplyClassifierOnlyTest)",
         base_dir / "ApplyClassifierOnlyTest" / "ApplyClassifierOnlyTest.py"),
    ]

    all_passed = True
    results = []

    for name, script in TEST_SCRIPTS:
        ok = run_test(name, script, cwd)
        results.append((name, ok))
        if not ok:
            all_passed = False

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for name, ok in results:
        status = "PASSED" if ok else "FAILED"
        print(f"{name}: {status}")

    print("=" * 70)
    if all_passed:
        print("✅All regression tests PASSED✅")
        sys.exit(0)
    else:
        print("⚠️ Some regression tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
