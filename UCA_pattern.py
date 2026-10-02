import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beamforming_uca import (c0, phi_m, kr_bin, choose_K, zl,
                              direct_beamform_uca, lc_beamform_uca_setup, lc_beamform_uca)

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

    K = choose_K(kr)
    
    beam_rad = np.arange(B) * 2 * np.pi / B

    n_sub = 20                        # fine-grid subdivisions per beam interval
    n_angles = B * n_sub             
    step_angle = 2 * np.pi / (B * n_sub)
    angles_rad = np.arange(n_angles) * step_angle

    theta0 = 2 * np.pi / B
    theta_beam = np.arange(B) * theta0

    setup = lc_beamform_uca_setup(M, kr, theta0, B, K)

    # direct-vs-low-complexity numerical validation at one representative steering angle
    x_check = np.exp(-1j * kr * np.cos(beam_rad[0] - phi))
    b_direct_check = direct_beamform_uca(x_check, kr, phi, M, theta0, B)

    b_lc_check = lc_beamform_uca(x_check, M, setup)
    relerr = np.max(np.abs(b_direct_check - b_lc_check)) / np.max(np.abs(b_direct_check))

    floor_db = -30.0
    # Beamforming
    own_gain_db = np.zeros((B, n_angles))
    for i, theta_in in enumerate(angles_rad):
        x_l = np.exp(-1j * kr * np.cos(theta_in - phi))  
        b_lc = lc_beamform_uca(x_l, M, setup)
    
        own_gain_db[:,i] = 20 * np.log10(np.abs(b_lc) + 1e-12)

    # Plotting  
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    cmap = plt.get_cmap("turbo")

    Error = np.zeros(B)
    # theta_dist = np.linspace(0, 2 * np.pi, 400)
    # ax.plot(theta_dist, np.full_like(theta_dist, -floor_db), "k--", lw=1.0,
    #     label="Distortionless (ideal), 0 dB")
    
    for i, theta_i in enumerate(theta_beam):
        pat_r = np.clip(own_gain_db[i, :], floor_db, 0.0) - floor_db
        ax.plot(angles_rad, pat_r, color=cmap(i / (B - 1)), lw=0.8)

        ax.plot(theta_i, pat_r[i*n_sub], linestyle="None", marker="o", ms=4,
                color="r", zorder=5, label="Steering direction" if i == 0 else None)
        Error[i] = abs(own_gain_db[i, int(theta_i//step_angle)]-0.0)

    
    ax.set_rlim(0, -floor_db)
    ax.set_rticks([0, 10, 20, 30])
    ax.set_yticklabels(["-30", "-20", "-10", "0 dB"])
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.legend(loc="upper right", framealpha=0.9, fontsize=6.5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(pad=0.3)
    fig.savefig("figs/uca_pattern.pdf")
    # plt.close(fig)
    plt.show()

    print(f"[Fig3-UCA] M={M}, K={K}, B={B}, kr={kr:.1f}, f0={f0/1e9:.2f} GHz")
    print(f"[Fig3-UCA] direct-vs-lc relative error (single beam, full grid) = {relerr:.2e}")
    print(f"[Fig3-UCA] own-angle gain (dB): mean={np.mean(Error):.4f}, "
          f"worst={np.max(Error):.4f}, best={np.min(Error):.4f} ")
         

if __name__ == "__main__":
    main()
