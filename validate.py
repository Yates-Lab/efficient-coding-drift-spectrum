"""
Validation tests. Run directly:  python validate.py

Checks:
  1. Lorentzian normalization: int dw/(2pi) C_D(k, w) == C_I(k) for each k.
  2. Rectification: |v*|^2 is zero exactly where C_D(k, w) <= C_th(lam).
  3. lambda solver: recovered P matches P_target.
  4. Whitening limit: in high-SNR, low-noise regime, |v*|^2 * C_x ~ const.
  5. Atick-Redlich duality spot check: at the same regime, the solution
     approaches 1/(lam * C_x).
"""

import numpy as np

from spectrum import C_D, C_I
from optimizer import active_threshold, solve_lambda, total_power, v_star_sq
from grids import build_radial_weights, omega_grid, radial_grid


def test_lorentzian_normalization():
    """
    Integral of C_D over omega should equal C_I(k). This requires resolving
    the Lorentzian at each k, whose half-width is D*k^2. For small k the
    peak is narrow; we pick a grid dense enough to resolve the narrowest
    peak tested, and wide enough to capture the tails of the widest.
    """
    print("== Test 1: Lorentzian normalization ==")
    D = 1.0
    A, beta = 1.0, 2.0
    ks = np.array([0.3, 0.5, 1.0, 2.0])

    # Narrowest half-width: D * ks.min()^2. Need dw << that.
    # Widest half-width: D * ks.max()^2. Need wmax >> that for tails.
    # Tail fraction beyond wmax is ~ (2/pi) * arctan(a/wmax) ~ 2a/(pi*wmax)
    # for wmax >> a. To get rel_err < 1e-4 we want wmax > 6500 * a_max.
    half_widths = D * ks**2
    dw_target = 0.05 * half_widths.min()
    wmax = 1e4 * half_widths.max()
    Nw = int(2 * wmax / dw_target) + 1
    # Ensure odd for symmetric endpoint inclusion.
    if Nw % 2 == 0:
        Nw += 1
    omega, dw_over_2pi = omega_grid(wmax, Nw)
    print(f"  grid: wmax={wmax:.1f}, Nw={Nw}, dw={omega[1]-omega[0]:.4g}")

    for k in ks:
        Cd = C_D(np.array([k]), omega, A=A, beta=beta, D=D, k0=0.0)
        integral = np.sum(Cd) * dw_over_2pi
        truth = C_I(np.array([k]), A=A, beta=beta, k0=0.0)[0]
        rel_err = abs(integral - truth) / truth
        print(f"  k={k:4.2f}  integral={integral:.6f}  truth={truth:.6f}  "
              f"rel_err={rel_err:.2e}")
        assert rel_err < 1e-3, f"Lorentzian normalization failed at k={k}"
    print("  PASSED\n")


def test_rectification():
    print("== Test 2: Rectification matches active threshold ==")
    kmax, Nk = 3.0, 200
    wmax, Nw = 20.0, 401
    k, _ = radial_grid(kmax, Nk, kmin=1e-3)
    omega, _ = omega_grid(wmax, Nw)

    Cd = C_D(k[:, None], omega[None, :], A=1.0, beta=2.0, D=1.0, k0=0.05)

    sigma_in_sq = 0.01
    sigma_out_sq = 0.01
    lam = 10.0  # Some value giving a nontrivial active set.

    v2 = v_star_sq(Cd, sigma_in_sq, sigma_out_sq, lam)
    Cth = active_threshold(sigma_in_sq, sigma_out_sq, lam)

    active_by_v = v2 > 0
    active_by_Cth = Cd > Cth

    # Allow tiny numerical disagreement on the boundary.
    disagreement = np.sum(active_by_v != active_by_Cth)
    total = active_by_v.size
    print(f"  C_th = {Cth:.6f}")
    print(f"  Active set size (v>0):  {active_by_v.sum()} / {total}")
    print(f"  Active set size (Cx>Cth): {active_by_Cth.sum()} / {total}")
    print(f"  Disagreement: {disagreement} cells  ({disagreement/total:.2%})")
    assert disagreement / total < 1e-3, "Rectification does not match threshold."
    print("  PASSED\n")


def test_lambda_solver():
    print("== Test 3: lambda solver recovers P_target ==")
    kmax, Nk = 3.0, 150
    wmax, Nw = 30.0, 401
    k, weight_k = radial_grid(kmax, Nk, kmin=1e-3)
    omega, dw_over_2pi = omega_grid(wmax, Nw)
    weights = build_radial_weights(weight_k, dw_over_2pi)

    Cd = C_D(k[:, None], omega[None, :], A=1.0, beta=2.0, D=1.0, k0=0.05)

    sigma_in_sq = 0.01
    sigma_out_sq = 0.01
    P_target = 0.5

    lam = solve_lambda(Cd, sigma_in_sq, sigma_out_sq, P_target, weights)
    v2 = v_star_sq(Cd, sigma_in_sq, sigma_out_sq, lam)
    P_check = total_power(Cd + sigma_in_sq, v2, weights)

    rel_err = abs(P_check - P_target) / P_target
    print(f"  lambda = {lam:.6g}")
    print(f"  P_target = {P_target:.6f}, P_check = {P_check:.6f}, "
          f"rel_err = {rel_err:.2e}")
    assert rel_err < 1e-6, "lambda solver failed."
    print("  PASSED\n")


def test_whitening_limit():
    print("== Test 4: Whitening in high-SNR, low-noise regime ==")
    # Full set of conditions for |v*|^2 ~ 1/(lam C_x):
    #   (a) C_x >> sigma_in^2
    #   (b) 4 sigma_in^2 / (lam sigma_out^2 C_x) << 1
    #   (c) lam sigma_out^2 << 1
    # Condition (c) is NOT stated in Section 2.5 of the notes but is needed:
    # the next-order expansion gives |v*|^2 ~ 1/(lam C_x) - sigma_out^2/C_x,
    # so the whitening term dominates only when 1/lam >> sigma_out^2.
    #
    # We pick (lam, sigma_in, sigma_out) directly to satisfy all three with
    # ~3 decades of safety, and use a wider grid to get enough cells.
    kmax, Nk = 3.0, 400
    wmax, Nw = 60.0, 801
    k, _ = radial_grid(kmax, Nk, kmin=0.02)
    omega, _ = omega_grid(wmax, Nw)

    Cd = C_D(k[:, None], omega[None, :], A=1.0, beta=2.0, D=1.0, k0=0.02)

    sigma_in_sq = 1e-4
    sigma_out_sq = 1.0
    lam = 1e-3

    v2 = v_star_sq(Cd, sigma_in_sq, sigma_out_sq, lam)
    active = v2 > 0
    cond_a = Cd > 100.0 * sigma_in_sq
    eps = 4.0 * sigma_in_sq / (lam * sigma_out_sq * Cd + 1e-30)
    cond_b = eps < 0.01
    mask = active & cond_a & cond_b
    assert lam * sigma_out_sq < 0.01, "Condition (c) not met."
    assert mask.sum() > 100, f"Only {mask.sum()} cells in whitening regime."

    pred = 1.0 / lam
    product = v2 * Cd
    vals = product[mask]
    rel_mean_err = abs(vals.mean() - pred) / pred
    rel_spread = (vals.max() - vals.min()) / pred
    print(f"  lam={lam}, sigma_in^2={sigma_in_sq}, sigma_out^2={sigma_out_sq}")
    print(f"  lam*sigma_out^2 = {lam*sigma_out_sq:.2e}")
    print(f"  Whitening-regime cells: {mask.sum()} / {active.sum()} active")
    print(f"  |v*|^2 * C_x: min={vals.min():.4g}, max={vals.max():.4g}, "
          f"mean={vals.mean():.4g}, predicted 1/lam={pred:.4g}")
    print(f"  Mean rel err: {rel_mean_err:.2%}, spread: {rel_spread:.2%}")
    assert rel_mean_err < 0.05, "Whitening mean off by > 5%."
    assert rel_spread < 0.10, "Whitening spread > 10%."
    print("  PASSED\n")


def test_scale_invariance_ideal():
    """
    Ideal d=2, beta=2 case: rescaling K = D^{1/4} k, Omega = D^{-1/2} omega
    makes C_D independent of D. We check C_D on the rescaled grid.
    """
    print("== Test 5: Scale invariance of C_D in the ideal case ==")
    K = np.linspace(0.1, 2.0, 5)
    Omega = np.linspace(0.1, 3.0, 5)

    ref = None
    for D in [0.1, 1.0, 10.0]:
        k = D ** (-0.25) * K
        omega = D ** 0.5 * Omega
        Cd = C_D(k[:, None], omega[None, :], A=1.0, beta=2.0, D=D, k0=0.0)
        if ref is None:
            ref = Cd
            ref_D = D
            continue
        rel_err = np.max(np.abs(Cd - ref) / ref)
        print(f"  D={D:>5g} vs D={ref_D:g}: max relative error = {rel_err:.2e}")
        assert rel_err < 1e-10, "Scale invariance broken for ideal case."
    print("  PASSED\n")


if __name__ == "__main__":
    test_lorentzian_normalization()
    test_rectification()
    test_lambda_solver()
    test_whitening_limit()
    test_scale_invariance_ideal()
    print("All tests passed.")
