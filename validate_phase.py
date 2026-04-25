"""
Test minimum-phase reconstruction against a known case.

Lorentzian amplitude: |v(omega)| = sqrt(a) / sqrt(a^2 + omega^2)
(so |v|^2 = a/(a^2 + omega^2)).

With the notes' Fourier convention f(t) = int dw/(2pi) exp(+i omega t) f(w),
the minimum-phase causal filter with this amplitude is
    v(t) = sqrt(a) * exp(-a t) * Theta(t),
whose frequency representation is
    v(w) = sqrt(a) / (a - i omega),
giving |v(w)| as above and phi(w) = -atan2(omega, a).

Compare our reconstructed phi(w) and v(t) to these truths.
"""

import numpy as np

from grids import omega_grid
from phase import ifft_to_time, minimum_phase


def main():
    a = 2.0
    wmax, Nw = 500.0, 200001
    omega, _ = omega_grid(wmax, Nw)

    v2 = a / (a**2 + omega**2)

    v_complex = minimum_phase(v2, floor_rel=1e-14, pad_factor=2)
    phi_recon = np.angle(v_complex)

    # True min phase (with the +i omega t convention).
    phi_true = -np.arctan2(omega, a)

    central = np.abs(omega) < wmax * 0.05
    err_abs = np.max(np.abs(phi_recon[central] - phi_true[central]))
    # scipy's discrete Hilbert imposes an implicit half-sample delay which
    # shows up as phase error linear in omega. Subtract a best-fit linear
    # term before checking the residual shape.
    diff = phi_recon[central] - phi_true[central]
    slope = np.polyfit(omega[central], diff, 1)[0]
    residual = diff - slope * omega[central]
    err_shape = np.max(np.abs(residual))
    print(f"Raw max phase error (central 10%): {err_abs:.4e}")
    print(f"Linear slope removed: {slope:.4e}  (~ dt/2 = {(np.pi/wmax)/2:.4e})")
    print(f"Shape error after slope removal: {err_shape:.4e}")
    assert err_shape < 0.01, "Phase shape wrong after slope removal."

    # Inverse transform to time domain.
    t, v_t = ifft_to_time(v_complex, omega)
    v_t_true = np.where(t >= 0, np.sqrt(a) * np.exp(-a * np.maximum(t, 0)), 0.0)

    # Compare away from the t=0 boundary. A one-sample-wide ringing at the
    # discontinuity in v_t_true is a generic artifact of representing a step
    # function on a finite omega grid (Gibbs); we exclude |t| < 2 dt.
    dt = t[1] - t[0]
    mask_t = (t > 2 * dt) & (t < 5.0 / a)
    err_t = np.max(np.abs(v_t[mask_t].real - v_t_true[mask_t]))
    peak = np.max(np.abs(v_t_true))
    print(f"Peak amplitude (truth): {peak:.4e}")
    print(f"Peak amplitude (recon): {np.max(np.abs(v_t.real)):.4e}")
    print(f"Max time-domain error (|t|>2dt window): {err_t:.4e} "
          f"({100*err_t/peak:.2f}% of peak)")
    print(f"Max |v_t| imag part: {np.max(np.abs(v_t.imag)):.4e}")

    # Causality: away from the t=0 jump, v(t) should be ~0 for t < 0.
    t_neg = (t < -2 * dt) & (t > -5.0 / a)
    max_leak = np.max(np.abs(v_t.real[t_neg]))
    print(f"Max precausal amplitude (t<-2dt): {max_leak:.4e} "
          f"({100*max_leak/peak:.2f}% of peak)")

    assert err_t / peak < 0.05, "Time-domain reconstruction inaccurate."
    assert max_leak / peak < 0.05, "Significant acausal leakage."
    print("PASSED")


if __name__ == "__main__":
    main()
