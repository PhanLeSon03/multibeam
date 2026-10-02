"""
Jacobi-Anger truncation error vs. Bessel order K.
"""
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beamforming import direct_beamform, choose_K, lc_beamform_setup, lc_beamform, zl, c0
from pattern_metrics import pattern_metrics, print_metrics, save_metrics_csv

rng = np.random.default_rng(0)


def main():
    THETA_LO_DEG, THETA_HI_DEG = -0.0, 60.0 
    PLOT_LO_DEG, PLOT_HI_DEG = -90.0, 90.0 

    N = 512
    M = 32
    B = 120
    fmin = 0.1e9
    fmax = 0.9e9
    fs = 2.0e9                  # ADC sample rate (bins land on multiples of fs/N)
    # f0 = fmax #0.9 * (fmin + fmax)    # representative in-band bin frequency for the pattern plot
    # l = round(f0 * N / fs)
    d = c0 / (2 * fs)

    l = N // 2 - 1         # near-Nyquist bin (worst-case aperture argument)
    z_ell = zl(d, N, l, fs)
    aperture_arg = (M - 1) * z_ell
    K_rule = choose_K(aperture_arg)

    m_vec = np.arange(M)
    n_sub = 40  # fine-grid subdivisions per beam interval
    step_deg = (THETA_HI_DEG - THETA_LO_DEG) / (B * n_sub)
    theta0= np.deg2rad(step_deg)*n_sub

    full_idx = round((PLOT_HI_DEG - PLOT_LO_DEG) / step_deg)
    angles_full_deg = PLOT_LO_DEG + np.arange(full_idx) * step_deg
    n_angles = full_idx
    

    # Original Beamforimg
    b_ref = np.zeros((B,n_angles), dtype=np.complex128)
    for i, theta_in in enumerate(angles_full_deg):
        x_ell = np.exp(-1j * z_ell * m_vec * np.sin(np.deg2rad(theta_in))) 
        b_ref[:, i] = direct_beamform(x_ell, z_ell, M, theta0, B)

    print(f"np.max(np.abs(b_ref)) = {np.max(np.abs(b_ref))}")

    K_range = np.arange(0, int(aperture_arg) + 32, 2)
    if K_rule not in K_range:
        K_range = np.sort(np.append(K_range, K_rule))
    metrics = []
    for K in K_range:
        b_k = np.zeros((B,n_angles), dtype=np.complex128)
        setup = lc_beamform_setup(M, z_ell, theta0, B, K)
        for i, theta_in in enumerate(angles_full_deg):
            x_ell = np.exp(-1j * z_ell * m_vec * np.sin(np.deg2rad(theta_in)))
            b_k[:, i] = lc_beamform(x_ell, M, setup) # (B,)

        metrics.append(pattern_metrics(b_k, b_ref, circular=False))
    errs = np.array([m["mbe"] for m in metrics])
    cplx = np.array([m["mce"] for m in metrics])
    integ = np.array([m["ibe"] for m in metrics])
    save_metrics_csv("figs/ULA_truncation_metrics.csv", K_range, metrics)
    print_metrics("ULA", K_rule, metrics[int(np.where(K_range == K_rule)[0][0])])

    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    ax.semilogy(K_range, errs, lw=1.3, color="C0", label="MBE")
    ax.semilogy(K_range, cplx, lw=1.1, color="C1", ls="--", label="MCE")
    ax.semilogy(K_range, integ, lw=1.1, color="C2", ls="-.", label="IBE")
    ax.axvline(aperture_arg, color="gray", lw=0.8, ls=":", label=r"$z_\ell$")
    ax.axvline(K_rule, color="C3", lw=0.8, ls="--", label=fr"chosen $K={K_rule}$")
    ax.set_xlabel(r"Truncation order $K$")
    ax.set_ylabel("Beam pattern error")
    # ax.set_ylim(1e-16, 2)
    ax.legend()
    fig.tight_layout(pad=0.3)
    fig.savefig("figs/ULA_truncation_error.pdf")
    # plt.close(fig)
    print(f"[Fig2] M={M}, aperture arg={aperture_arg:.1f}, rule-of-thumb K={K_rule}")
    plt.show()


if __name__ == "__main__":
    main()
