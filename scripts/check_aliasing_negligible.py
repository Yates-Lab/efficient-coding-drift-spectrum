"""Sanity check: aliasing is negligible for our band parameters.

The appendix's main derivation assumes the m=0 copy of the spectrum
dominates over folded copies. This holds when the input spectrum has
little power at spatial frequencies above the Nyquist of the mosaic,
*at frequencies that contribute substantially to the optimization*.

For a mosaic critically sampled at f_max (so k_s = 2 f_max), aliased
copies arrive at radii f_max, 3 f_max, 5 f_max, ... For our band
(f_max = 4 cyc/u), the first folded copy lives at 3 f_max = 12 cyc/u.

We compute the power-weighted ratio: how much of the total in-band
spectral power would be added by aliased copies, relative to the direct
copy. This is the quantity that actually matters for the optimization,
not the pointwise ratio at the band edge.
"""

import sys
sys.path.insert(0, ".")

import numpy as np

from src.spectra import drift_spectrum
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX


def main():
    f = np.geomspace(0.05, F_MAX, 80)
    omega = np.geomspace(OMEGA_MIN, OMEGA_MAX, 80)
    F, W = np.meshgrid(f, omega, indexing="ij")

    D_values = [0.05, 0.5, 5.0, 50.0]
    beta = 2.0

    df = np.gradient(f)[:, None]
    domega = np.gradient(omega)[None, :]
    weights = 2 * np.pi * F * df * domega

    print(f"Band: f <= {F_MAX} cyc/u, {OMEGA_MIN} <= |omega| <= {OMEGA_MAX} rad/s")
    print()

    for oversample in [1.0, 1.5, 2.0]:
        k_s = 2 * F_MAX * oversample
        print(f"=== Mosaic with k_s = {oversample:.1f} x (2 f_max) = {k_s:.1f} cyc/u ===")
        print(f"  {'D':>6}  {'direct power':>14}  {'aliased power':>14}  {'ratio':>10}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*10}")

        for D in D_values:
            C_direct = drift_spectrum(F, W, D=D, beta=beta)
            C_alias = np.zeros_like(C_direct)
            for m in range(-2, 3):
                for n in range(-2, 3):
                    if m == 0 and n == 0:
                        continue
                    f_alias = np.sqrt((F + m * k_s) ** 2 + (n * k_s) ** 2)
                    C_alias = C_alias + drift_spectrum(f_alias, W, D=D, beta=beta)

            P_direct = np.sum(C_direct * weights)
            P_alias = np.sum(C_alias * weights)
            ratio = P_alias / max(P_direct, 1e-30)
            print(f"  {D:6.2f}  {P_direct:14.4e}  {P_alias:14.4e}  {ratio:10.3f}")
        print()

    print("Interpretation: 'ratio' is the fraction of the in-band signal power")
    print("contributed by aliased copies. With critical sampling (1.0x), aliasing")
    print("is significant at low D. With moderate over-sampling (1.5-2.0x), the")
    print("m=0 dominant-copy approximation becomes well justified.")
    print()
    print("Our figures use the direct spectrum (no aliasing), which corresponds")
    print("to the limit of fine over-sampling, or equivalently to assuming the")
    print("mosaic spacing is set by something other than f_max (e.g., the")
    print("photoreceptor sampling rate, which in primates is much finer than")
    print("the cells whose RFs we're modelling).")


if __name__ == "__main__":
    main()
