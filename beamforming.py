"""
Implements, from first principles, both:
  (1) the direct per-bin multibeam DS beamformer -- O(M B) per frequency bin
  (2) the proposed low-complexity beamformer: Jacobi-Anger (Bessel) expansion
      + a chirp-Z transform (Bluestein's algorithm) -- O(M K + (K+B)log(K+B))
      per frequency bin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.signal import CZT

c0 = 3e8

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "legend.fontsize": 8,
})


def zl(d, N, l, fs, c=c0):
    """Electrical spacing z = 2*pi*d*f/c at a frequency f (rad)."""
    return 2 * np.pi * d * fs * l / c / N

def z_bin(d, f, c=c0):
    """Electrical spacing z = 2*pi*d*f/c at a frequency f (rad)."""
    return 2 * np.pi * d * f / c


def direct_beamform(x_ell, z_ell, M, theta0, B):
    """Direct per-bin multibeam DS: b[i] = (1/M) sum_m x[m] exp(j z m sin(i theta0)).
    Cost O(MB) once the (M,B) steering matrix is formed."""
    m = np.arange(M)
    i = np.arange(B)
    W = np.exp(1j * z_ell * np.outer(m, np.sin(i * theta0))) / M  # (M,B)
    return x_ell @ W


def choose_K(aperture_arg, target_db=-30):
    """Smallest truncation order K such that |J_K(aperture_arg)| is below
    target_db, i.e. the Jacobi-Anger tail beyond +-K is numerically negligible
    for every element m (whose largest Bessel argument is (M-1)*z_ell)."""
    K = int(np.ceil(aperture_arg)) + 1
    while True:
        if abs(jv(K, aperture_arg)) < 10 ** (target_db / 20):
            return K
        K += 1


def lc_beamform_setup(M, z_l, theta0, B, K, theta_start=0.0):
    
    m_vec = np.arange(M)
    ks = np.arange(-K, K + 1)
    Jmat = np.zeros((2 * K + 1, M), dtype=np.complex128) 
    for k in ks:
        Jmat[k + K, :] = jv(k, m_vec * z_l)

    czt = CZT(n=2 * K + 1, m=B, w=np.exp(1j * theta0), a=np.exp(-1j * theta_start))
    # CZT input index n = k + K, so undo the shift: e^{-jK(theta_start + i*theta0)}
    phase_out = np.exp(-1j * K * (theta_start + theta0 * np.arange(B)))

    return Jmat, czt, phase_out


def lc_beamform(x_ell, M, setup):
    """mutlibeam DS via Jacobi-Anger expansion + chirp-Z transform."""
    Jmat, czt, phase_out = setup
    y_ell = (Jmat @ x_ell) / M # (K,M)x(M,) -> (K,)
    return phase_out * czt(y_ell)  # (B,)
