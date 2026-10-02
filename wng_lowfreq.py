import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beamforming import choose_K, zl, c0
from beamforming_uca import kr_bin
from ULA_wng import ula_wng_vs_K, uca_wng_vs_K, _rep_indices


def main():
    # ---- (a) ULA at fmin, same array as fig2/fig9 ----
    N, M = 512, 32
    fs = 2.0e9
    d = c0 / (2 * fs)
    fmin = 0.1e9
    l = int(np.ceil(fmin * N / fs))
    fd = fs * l / N        # actual bin frequency, for the panel title



    z_ell2 = zl(d, N, l, fs)
    aperture_arg2 = (M - 1) * abs(z_ell2)
    K2_rule = choose_K(aperture_arg2)

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
    TARGET_KR = 5.0*(0.1 / 0.9)
    l_max = N//2 -1
    f_max = fs * l_max / N  

    r = TARGET_KR * c0 / (2 * np.pi * f_max)
     

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
    ax1.set_title(f"(a) ULA, $f={fd/1e9:.2f}$ GHz", fontsize=9)
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
    ax2.set_title(f"(b) UCA,  $f={fd/1e9:.2f}$ GHz", fontsize=9)
    ax2.legend(fontsize=5, loc="lower right", ncol=2)

    fig.tight_layout(pad=0.3)
    fig.savefig("figs/fig11_wng_lowfreq.pdf", bbox_inches="tight")
    for th_deg in theta_i_deg_ula:
        g_at_rule = 10 * np.log10(ula_wng_vs_K(M, z_ell2, [K2_rule], theta_i=np.deg2rad(th_deg)))[0]
        print(f"[Fig11-WNG-lowfreq] ULA: theta_i={th_deg:.1f} deg, M={M}, aperture arg={aperture_arg2:.1f}, "
              f"K_rule={K2_rule}, G_wn(K_rule)={g_at_rule:.2f} dB (ideal {ideal_ula_db:.2f} dB)")
    for th_deg in theta_i_deg_uca:
        g_at_rule = 10 * np.log10(uca_wng_vs_K(M, kr, [K7_rule], theta_i=np.deg2rad(th_deg)))[0]
        print(f"[Fig11-WNG-lowfreq] UCA: theta_i={th_deg:.0f} deg, M={M}, kr={kr:.1f}, "
              f"K_rule={K7_rule}, G_wn(K_rule)={g_at_rule:.2f} dB (ideal {ideal_uca_db:.2f} dB)")
    plt.show()


if __name__ == "__main__":
    main()


