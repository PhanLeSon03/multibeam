"""
Extension of the low-complexity multibeam DS beamformer to uniform circular
arrays (UCAs).
"""
import numpy as np
from scipy.special import jv
from scipy.signal import CZT

from beamforming import c0, zl, z_bin, choose_K  


def phi_m(M):
    return 2 * np.pi * np.arange(M) / M


def kr_bin(r, f, c=c0):
    """Electrical radius kr = 2*pi*r*f/c at a frequency f (rad). Same formula
    as the ULA's electrical spacing z_bin, since kr and z=k*d share the
    identical 2*pi*(.)*f/c form -- only the geometric meaning differs."""
    return z_bin(r, f, c)


def direct_beamform_uca(x_ell, kr_ell, phi_ell, M, theta0, B):
    """Direct per-bin multibeam DS for a UCA:
    b[i] = (1/M) sum_m x[m] exp(j kr cos(i*theta0 - phi_m)).
    Cost O(MB) once the (M,B) steering matrix is formed."""
    i_vec = np.arange(B)
    psi = i_vec * theta0
    Psi = kr_ell * np.cos(psi[None, :] - phi_ell[:, None])  # (M,B)
    W = np.exp(1j * Psi) / M
    return x_ell @ W


def lc_beamform_uca_setup(M, kr_ell, theta0, B, K):
  
    n = np.arange(-K, K + 1)
    Jn = (1j ** n) * jv(n, kr_ell)  # j^n J_n(kr), length 2K+1

    czt = CZT(n=2 * K + 1, m=B, w=np.exp(1j * theta0))
    phase_out = np.exp(-1j * theta0 * np.arange(B) * K)
    return Jn, czt, phase_out


def lc_beamform_uca(x_ell, M, setup):
    Jn, czt, phase_out = setup
    K = (len(Jn) - 1) // 2
    Xn_full = np.fft.fft(x_ell) / M             
    idx = np.arange(-K, K + 1) % M              
    y = Jn * Xn_full[idx]
    return phase_out * czt(y)
