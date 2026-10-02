"""
White noise gain (WNG) vs. truncation order K, for both array types 
"""
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jv

from beamforming import choose_K, zl, c0
from beamforming_uca import phi_m, kr_bin


def _rep_indices(B):
    return [0, (B - 1) // 4, (B - 1) // 2, 3 * (B - 1) // 4, B - 1]


def ula_wng_vs_K(M, z_ell, K_range, theta_i=0.0):
    m_vec = np.arange(M)
    a = np.exp(1j * z_ell * m_vec * np.sin(theta_i))  # true steering vector
    G = np.zeros(len(K_range))
    for idx, K in enumerate(K_range):
        ks = np.arange(-K, K + 1)
        Jk = jv(ks[:, None], (m_vec * z_ell)[None, :])  # (2K+1, M)
        phase = np.exp(1j * ks * theta_i)  # (2K+1,)
        S = (phase[:, None] * Jk).sum(axis=0)  # (M,)
        G[idx] = np.abs(np.sum(np.conj(S) * a)) ** 2 / np.sum(np.abs(S) ** 2)
    return G


def uca_wng_vs_K(M, kr, K_range, theta_i=0.0):
    phi = phi_m(M)
    a = np.exp(1j * kr * np.cos(theta_i - phi))  # true steering vector
    G = np.zeros(len(K_range))
    for idx, K in enumerate(K_range):
        ns = np.arange(-K, K + 1)
        Jn = (1j ** ns) * jv(ns, kr)  # (2K+1,)
        phase = np.exp(1j * ns * theta_i)  # (2K+1,)
        T = ((Jn * phase)[:, None] * np.exp(-1j * np.outer(ns, phi))).sum(axis=0)  # (M,)
        G[idx] = np.abs(np.sum(np.conj(T) * a)) ** 2 / np.sum(np.abs(T) ** 2)
    return G


def main():
    # ---- (a) ULA: match fig2_truncation.py exactly ----
    N, M = 512, 32
    fs = 2.0e9
    d = c0 / (2 * fs)
    l = N // 2 - 1         # near-Nyquist bin (worst-case aperture argument)
    fd = fs * l / N        # actual bin frequency, for the panel title

    z_ell2 = zl(d, N, l, fs)
    aperture_arg2 = (M - 1) * abs(z_ell2)
    K2_rule = choose_K(aperture_arg2)

    # Representative incident directions across the same swept sector used in
    # fig1_pattern.py / fig12_unit_gain_coverage.py / fig13.
    B = 120
    THETA_LO_DEG, THETA_HI_DEG = 0.0, 60.0
    beam_spacing_deg = (THETA_HI_DEG - THETA_LO_DEG) / B
    reps = _rep_indices(B)
    theta_i_deg_ula = [THETA_LO_DEG + i * beam_spacing_deg for i in reps]

    K_range_ula = np.arange(0, int(aperture_arg2) + 40, 2)
    G_ula_db = np.zeros((len(theta_i_deg_ula), len(K_range_ula)))
    for r_idx, th_deg in enumerate(theta_i_deg_ula):
        G = ula_wng_vs_K(M, z_ell2, K_range_ula, theta_i=np.deg2rad(th_deg))
        G_ula_db[r_idx] = 10 * np.log10(np.clip(G, 1e-6, None))

    # ---- (b) UCA: match fig7_uca_truncation.py exactly ----
    TARGET_KR = 5.0
    
    r = TARGET_KR * c0 / (2 * np.pi * fd)

    print(f"r={r}")
    kr = zl(r, N, l, fs)
    K7_rule = choose_K(kr)
   
    beam_spacing_deg_uca = 360.0 / B
    theta_i_deg_uca = [i * beam_spacing_deg_uca for i in reps]

    K_range_uca = np.arange(0, K7_rule + 10)
    G_uca_db = np.zeros((len(theta_i_deg_uca), len(K_range_uca)))
    for r_idx, th_deg in enumerate(theta_i_deg_uca):
        G = uca_wng_vs_K(M, kr, K_range_uca, theta_i=np.deg2rad(th_deg))
        G_uca_db[r_idx] = 10 * np.log10(np.clip(G, 1e-6, None))

    ideal_ula_db = 10 * np.log10(M)
    ideal_uca_db = 10 * np.log10(M)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))

    colors = [f"C{k}" for k in range(len(reps))]

    for k, th_deg in enumerate(theta_i_deg_ula):
        ax1.plot(K_range_ula, G_ula_db[k], lw=1.1, color=colors[k],
                  label=fr"$\theta_i={th_deg:.1f}\degree$")
    ax1.axhline(ideal_ula_db, color="k", lw=0.8, ls="-.", label=fr"ideal $10\log_{{10}}M={ideal_ula_db:.1f}$ dB")
    ax1.axvline(aperture_arg2, color="gray", lw=0.8, ls=":", label=r"$z_\ell$")
    ax1.axvline(K2_rule, color="C3", lw=0.8, ls="--", label=fr"chosen $K={K2_rule}$")
    ax1.set_xlabel(r"Truncation order $K$")
    ax1.set_ylabel(r"WNG (dB)")
    ax1.set_ylim(-5, ideal_ula_db + 3)
    ax1.set_title(f"(a) ULA, $f={fd/1e9:.3f}$ GHz", fontsize=9)
    ax1.legend(fontsize=5, loc="lower right", ncol=2)

    for k, th_deg in enumerate(theta_i_deg_uca):
        ax2.plot(K_range_uca, G_uca_db[k], lw=1.1, color=colors[k],
                  label=fr"$\theta_i={th_deg:.0f}\degree$")
    ax2.axhline(ideal_uca_db, color="k", lw=0.8, ls="-.", label=fr"ideal $10\log_{{10}}M={ideal_uca_db:.1f}$ dB")
    ax2.axvline(kr, color="gray", lw=0.8, ls=":", label=r"$z_\ell$")
    ax2.axvline(K7_rule, color="C3", lw=0.8, ls="--", label=fr"chosen $K={K7_rule}$")
    ax2.set_xlabel(r"Truncation order $K$")
    ax2.set_ylabel(r"WNG (dB)")
    ax2.set_ylim(-5, ideal_uca_db + 3)
    ax2.set_title(f"(b) UCA,  $f={fd/1e9:.3f}$ GHz", fontsize=9)
    ax2.legend(fontsize=5, loc="lower right", ncol=2)

    fig.tight_layout(pad=0.3)
    fig.savefig("figs/fig9_wng.pdf", bbox_inches="tight")
    for th_deg in theta_i_deg_ula:
        g_at_rule = 10 * np.log10(ula_wng_vs_K(M, z_ell2, [K2_rule], theta_i=np.deg2rad(th_deg)))[0]
        print(f"[Fig9-WNG] ULA: theta_i={th_deg:.1f} deg, M={M}, aperture arg={aperture_arg2:.1f}, "
              f"K_rule={K2_rule}, G_wn(K_rule)={g_at_rule:.2f} dB (ideal {ideal_ula_db:.2f} dB)")
    for th_deg in theta_i_deg_uca:
        g_at_rule = 10 * np.log10(uca_wng_vs_K(M, kr, [K7_rule], theta_i=np.deg2rad(th_deg)))[0]
        print(f"[Fig9-WNG] UCA: theta_i={th_deg:.0f} deg, M={M}, kr={kr:.1f}, "
              f"K_rule={K7_rule}, G_wn(K_rule)={g_at_rule:.2f} dB (ideal {ideal_uca_db:.2f} dB)")
    plt.show()


if __name__ == "__main__":
    main()
