import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beamforming import z_bin, choose_K, lc_beamform_setup, lc_beamform, c0, zl

M = 32
N = 512
B = 120                      # number of simultaneous beams
THETA_LO_DEG, THETA_HI_DEG = -0.0, 60.0   # testing range: beams whose own-angle gain is checked
PLOT_LO_DEG, PLOT_HI_DEG = -90.0, 90.0     # working angle: full broadside-referenced field of view


def main():
    fmin = 0.1e9
    fmax = 0.9e9
    fs = 2.0e9                  # ADC sample rate (bins land on multiples of fs/N)
    # f0 = fmax #0.9 * (fmin + fmax)    # representative in-band bin frequency for the pattern plot
    # l = round(f0 * N / fs)
    l = N // 2 - 1   # near-Nyquist bin (worst-case aperture argument)
    d = c0 / (2 * fs)

    print(f"inter-distance:{d}, array size: {(M-1)*d} ")

    z_l = zl(d, N, l, fs)
    f_bin = fs * l / N
    m_vec = np.arange(M)

    aperture_arg = (M - 1) * z_l
    K = choose_K(aperture_arg)     # Bessel truncation order set from the actual in-band aperture arg
    beam_deg = THETA_LO_DEG + np.arange(B) * (THETA_HI_DEG - THETA_LO_DEG) / (B - 1)

    n_sub = 40  # fine-grid subdivisions per beam interval
    step_deg = (THETA_HI_DEG - THETA_LO_DEG) / ((B - 1) * n_sub)
    full_idx = round((PLOT_HI_DEG - PLOT_LO_DEG) / step_deg)
    
    angles_full_deg = PLOT_LO_DEG + np.arange(full_idx + 1) * step_deg
    angles_full_rad = np.deg2rad(angles_full_deg)

    theta0= np.deg2rad(step_deg)*n_sub
    n_angles = full_idx + 1  
    
    setup = lc_beamform_setup(M, z_l, theta0, B, K)

    
    floor_db = -30.0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    cmap = plt.get_cmap("turbo")
    own_gain_db = np.zeros((B, n_angles))
    
    # Beamforimg
    for i, theta_in in enumerate(angles_full_deg):
        x_l = np.exp(-1j * z_l * m_vec * np.sin(np.deg2rad(theta_in)))  
        b_l = lc_beamform(x_l, M, setup) # (B,)
        pat_db_full = 20 * np.log10(np.abs(b_l) + 1e-12)
        own_gain_db[:,i] = pat_db_full
        
    # Plotting  
    Error = np.zeros(B)
    for i, theta_i in enumerate(beam_deg):
        pat_r = np.clip(own_gain_db[i, :], floor_db, 0.0) - floor_db
        ax.plot(angles_full_rad, pat_r, color=cmap(i / (B - 1)), lw=0.8)

        # Mark the steering direction with a red dot
        ax.plot(np.deg2rad(theta_i), pat_r[int((theta_i-PLOT_LO_DEG)//step_deg)], linestyle="None", marker="o", ms=4.0,
         color="r", zorder=5, label="Steering direction" if i == 0 else None)
        
        Error[i] = abs(own_gain_db[i,int((theta_i-PLOT_LO_DEG)//step_deg)] - 0.0)
    

    ax.set_theta_zero_location("N")   # broadside (0 deg) points up
    ax.set_theta_direction(-1)        # positive angles to the right
    ax.set_thetamin(PLOT_LO_DEG)
    ax.set_thetamax(PLOT_HI_DEG)
    ax.set_rlim(0, -floor_db)
    ax.set_rticks([0, 10, 20, 30])
    ax.set_yticklabels(["-30", "-20", "-10", "0 dB"])
    ax.set_xticks(np.deg2rad([-90, -60, -30, 0, 30, 60, 90]))
    ax.legend(loc="center", framealpha=0.9, fontsize=6.5, bbox_to_anchor=(0.5, 0.1))
    fig.tight_layout(pad=0.3)
    fig.savefig("figs/multibeam_pattern.pdf")
    # plt.close(fig)
    plt.show()

    print(f"[Fig1] B={B}, beam_deg={len(beam_deg)}, n_angles={n_angles}, theta in [{THETA_LO_DEG},{THETA_HI_DEG}] deg ")
    print(f"[Fig1] M={M}, K={K},"
          f"f_bin={f_bin/1e9:.3f} GHz (in [{fmin/1e9:.2f},{fmax/1e9:.2f}] GHz), "
          f"aperture arg={aperture_arg:.1f}")
    print(f"[Fig1] Max error at steering angles: {np.max(Error):.2e} dB,")
    print(f"[Fig1] Min error at steering angles: {np.min(Error):.2e} dB")
    print(f"[Fig1] Mean error at steering angles: {np.mean(Error):.2e} dB")

         


if __name__ == "__main__":
    main()
