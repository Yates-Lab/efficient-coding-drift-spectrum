#%%
"""
Run all validation tests and generate all four figures.

Order:
  1. validate.py              - optimizer & spectrum tests
  2. validate_phase.py        - minimum-phase reconstruction tests
  3. fig1_filter_magnitude.py - |v*_D(k, omega)|^2 heatmaps
  4. fig2_temporal_kernels.py - v*_D(k, t) at representative k
  5. fig3_info_vs_drift.py    - I(D) curve with D* and fixed-filter baselines
  6. fig4_scaling.py          - empirical k_max, omega_max scaling
"""

import subprocess
import sys

SCRIPTS = [
    "validate.py",
    "validate_phase.py",
    "fig1_input_and_filter.py",
    "fig2_temporal_kernels.py",
    "fig3_info_vs_drift.py",
    "fig4_scaling.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'='*60}\n== {script}\n{'='*60}")
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"\n{script} failed with code {result.returncode}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()

# %%
