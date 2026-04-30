"""Run all tests and generate all figures."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

print("=" * 60)
print("Running tests")
print("=" * 60)
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v"],
    cwd=ROOT,
)
if result.returncode != 0:
    print("Tests failed.")
    sys.exit(1)

print("\n" + "=" * 60)
print("Generating figures")
print("=" * 60)
for fig in [
    "figures/fig1_power_spectra.py",
    "figures/fig2_optimal_filter.py",
    "figures/fig3_kernels.py",
    "figures/fig4_information_vs_D.py",
    "figures/fig5_kernel_slices.py",
]:
    print(f"\n>>> {fig}")
    r = subprocess.run([sys.executable, fig], cwd=ROOT)
    if r.returncode != 0:
        print(f"Failed: {fig}")
        sys.exit(1)

print("\n" + "=" * 60)
print("Done. Outputs in outputs/.")
print("=" * 60)
