"""
One-time, local-only sanity check: confirms metrics.math_utils.seg_score_volume
(the native Python reimplementation of the CTC SEG formula) matches the official
./SEGMeasure binary's output on a real experiment folder.

This is NOT part of the training/testing pipeline -- the whole point of the native
reimplementation is to remove the runtime dependency on SEGMeasure (which doesn't
work on the DFKI Pegasus cluster). Run this manually, locally, where the binary
still works, to build confidence in the native implementation before trusting it
on the cluster.

Usage:
    python utils/validate_seg_against_binary.py <exp_dir> [<exp_dir> ...]

Each <exp_dir> must already have 01_GT/SEG/man_seg*.tif and 01_RES/mask*.tif
populated (e.g. by a prior testing_loop.py or training_loop.py run).
"""
import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emrConfigManager import REPO_ROOT
from metrics.metrics_volume import emrMetricsVolume

TOLERANCE = 1e-4


def validate(exp_dir: str) -> bool:
    exp_dir = os.path.abspath(exp_dir)
    print(f"\n=== {exp_dir} ===")

    binary_result = subprocess.run(
        [str(REPO_ROOT / "SEGMeasure"), exp_dir, "01", "4"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    stdout = binary_result.stdout.strip()
    print(f"SEGMeasure stdout: {stdout!r}")
    try:
        binary_score = float(stdout[stdout.find(":") + 1:])
    except ValueError:
        print("FAILED: could not parse a SEG score out of SEGMeasure's stdout.")
        return False

    native_score = emrMetricsVolume(exp_dir).seg_score()["mean"]

    diff = abs(binary_score - native_score)
    print(f"SEGMeasure (binary):     {binary_score:.6f}")
    print(f"seg_score_volume (native): {native_score:.6f}")
    print(f"abs diff:                {diff:.6f} (tolerance {TOLERANCE})")

    ok = diff <= TOLERANCE
    print("MATCH" if ok else "MISMATCH -- investigate before trusting the native implementation")
    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python utils/validate_seg_against_binary.py <exp_dir> [<exp_dir> ...]")

    results = [validate(exp_dir) for exp_dir in sys.argv[1:]]
    print(f"\n{sum(results)}/{len(results)} experiment folders matched within tolerance.")
    if not all(results):
        sys.exit(1)
