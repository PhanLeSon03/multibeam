import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beamforming_uca import (c0, phi_m, zl, choose_K,
                              direct_beamform_uca, lc_beamform_uca_setup, lc_beamform_uca)
from pattern_metrics import pattern_metrics, print_metrics, save_metrics_csv


M = 32
B = 120                     # number of simultaneous beams around the circle
TARGET_KR = 5.              # electrical radius at the representative frequency
N = 512

def main():
    fs = 2e9
    l = N // 2 - 1  
    f0 = fs*l/N
    r = TARGET_KR * c0 / (2 * np.pi * f0)
    
   
    kr = zl(r, N, l, fs, c=c0)
    print(f"r={r}, kr = {kr}")

    phi = phi_m(M)

    K_rule = choose_K(kr)

    n_sub = 20
    step_rad = 2*np.pi / (B* n_sub)
    theta0 = step_rad * n_sub      # = 2*pi/B; step_rad is already in radians
    
    full_idx = round(2*np.pi/ step_rad)
    angles_full_deg = np.arange(full_idx) * step_rad
    n_angles = full_idx

    # Original Beamforimg
    b_ref = np.zeros((B,n_angles), dtype=np.complex128)
    for i, theta_in in enumerate(angles_full_deg):
        x_l = np.exp(-1j * kr * np.cos(theta_in - phi)) 
        b_ref[:, i] = direct_beamform_uca(x_l, kr, phi, M, theta0, B)

    K_range = np.arange(0, K_rule + 7)
    metrics = []
    for K in K_range:
        b_k = np.zeros((B,n_angles), dtype=np.complex128)
        setup = lc_beamform_uca_setup(M, kr, theta0, B, K)
        for i, theta_in in enumerate(angles_full_deg):
            x_l = np.exp(-1j * kr * np.cos(theta_in - phi))
            b_k[:, i]  = lc_beamform_uca(x_l, M, setup)

        metrics.append(pattern_metrics(b_k, b_ref, circular=True))
    errs = np.array([m["mbe"] for m in metrics])
    cplx = np.array([m["mce"] for m in metrics])
    integ = np.array([m["ibe"] for m in metrics])
    save_metrics_csv("figs/uca_truncation_metrics.csv", K_range, metrics)
    print_metrics("UCA-Truncation", K_rule, metrics[K_rule])

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.semilogy(K_range, errs + 1e-16, lw=1.3, color="C0", label="MBE")
    ax.semilogy(K_range, cplx + 1e-16, lw=1.1, color="C1", ls="--", label="MCE")
    ax.semilogy(K_range, integ + 1e-16, lw=1.1, color="C2", ls="-.", label="IBE")
    ax.axvline(kr, color="gray", lw=0.8, ls=":", label=r"$z_\ell$")
    ax.axvline(K_rule, color="C3", lw=0.8, ls="--", label=fr"chosen $K={K_rule}$")
    ax.set_xlabel(r"Truncation order $K$")
    ax.set_ylabel("Beam pattern error")
    # ax.set_ylim(1e-16, 2)
    ax.legend()
    fig.tight_layout(pad=0.3)
    fig.savefig("figs/uca_truncation_error.pdf")
    # plt.close(fig)
    print(f"[Fig7-UCA] M={M}, kr={kr:.1f}, rule-of-thumb K={K_rule}")
    plt.show()


if __name__ == "__main__":
    main()
