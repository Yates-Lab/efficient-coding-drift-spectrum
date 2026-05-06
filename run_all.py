"""Run all tests and generate all figures.

Use ``--with-cell-learning`` for the production cell-class artifacts. That
path calls ``scripts/run_cell_class_learning.py``, whose default optimizer is
the fast oracle-initialized implementation.
"""

import argparse

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--skip-tests", action="store_true")
parser.add_argument("--skip-figures", action="store_true")
parser.add_argument("--with-cell-learning", action="store_true",
                    help="also run the fast production cell-class fit")
parser.add_argument("--with-cell-story", action="store_true",
                    help="also generate cell-class story figures; implies --with-cell-learning")
parser.add_argument("--cell-outdir", default="outputs/cell_classes")
parser.add_argument("--cell-story-outdir", default="outputs/cell_classes_story")
parser.add_argument("--cell-kmax", type=int, default=4)
parser.add_argument("--cell-steps", type=int, default=1600)
parser.add_argument("--cell-restarts", type=int, default=2)
parser.add_argument("--cell-device", default="auto")
parser.add_argument("--cell-dtype", default="float32", choices=("float32", "float64"))
args = parser.parse_args()

if args.with_cell_story:
    args.with_cell_learning = True


def run_checked(cmd, *, label):
    print(f"\n>>> {label}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"Failed: {label}")
        sys.exit(1)


def run_figure(path):
    """Run a figure file, allowing IPython-style magic lines in scripts."""
    bootstrap = (
        "from pathlib import Path\n"
        "import sys\n"
        "path = Path(sys.argv[1])\n"
        "source = '\\n'.join(\n"
        "    line for line in path.read_text().splitlines()\n"
        "    if not line.lstrip().startswith('%')\n"
        ")\n"
        "ns = {'__file__': str(path), '__name__': '__main__'}\n"
        "exec(compile(source, str(path), 'exec'), ns)\n"
    )
    run_checked([sys.executable, "-c", bootstrap, path], label=path)


if not args.skip_tests:
    print("=" * 60, flush=True)
    print("Running tests", flush=True)
    print("=" * 60, flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("Tests failed.")
        sys.exit(1)

if not args.skip_figures:
    print("\n" + "=" * 60, flush=True)
    print("Generating figures", flush=True)
    print("=" * 60, flush=True)
    for fig in [
        "figures/fig1_power_spectra.py",
        "figures/fig2_optimal_filter.py",
        "figures/fig3_kernels.py",
        "figures/fig4_information_vs_D.py",
        "figures/fig5_kernel_slices.py",
        "figures/fig6_saccade_kernels.py",
        "figures/fig6b_saccade_diagnostics.py",
        "figures/fig6c_saccade_vs_drift_kernels.py",
        "figures/fig7_rucci_cycle_spectra.py",
        "figures/fig8_mostofi_saccade_approximation.py",
        "figures/figQ1_spectrum_library.py",
        "figures/figQ2_information_sweeps.py",
        "figures/figQ3_magno_parvo.py",
    ]:
        run_figure(fig)

if args.with_cell_learning:
    print("\n" + "=" * 60, flush=True)
    print("Running fast cell-class learning", flush=True)
    print("=" * 60, flush=True)
    run_checked(
        [
            sys.executable,
            "scripts/run_cell_class_learning.py",
            "--optimizer", "fast",
            "--grid", "fast",
            "--kmax", str(args.cell_kmax),
            "--steps", str(args.cell_steps),
            "--restarts", str(args.cell_restarts),
            "--device", args.cell_device,
            "--dtype", args.cell_dtype,
            "--outdir", args.cell_outdir,
        ],
        label="scripts/run_cell_class_learning.py",
    )

if args.with_cell_story:
    print("\n" + "=" * 60, flush=True)
    print("Generating cell-class story figures", flush=True)
    print("=" * 60, flush=True)
    run_checked(
        [
            sys.executable,
            "scripts/make_cell_class_story_figures.py",
            "--indir", args.cell_outdir,
            "--outdir", args.cell_story_outdir,
            "--K", "2",
        ],
        label="scripts/make_cell_class_story_figures.py",
    )

print("=" * 60, flush=True)
print("Done. Outputs in outputs/.", flush=True)
print("=" * 60, flush=True)
