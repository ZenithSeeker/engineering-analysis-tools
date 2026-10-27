"""What frequency is the 'wub-wub' you get driving with one window down?

Recorded on a phone in a moving car (data/nz_car_window_buffeting.m4a). With one window open the
cabin behaves like a Helmholtz resonator: the air in the opening is a mass bouncing on the spring
of the cabin air, and the shear layer over the opening locks onto that resonance.

This finds the tone in the recording, then compares it against a Helmholtz estimate.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, spectrogram, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fft_analysis import load_audio

# ---------- Helmholtz estimate. ASSUMED values for a small hatchback: replace with measurements
C = 343.0               # speed of sound, m/s
V_CABIN = 2.8           # cabin air volume, m^3
A_OPEN = 0.45 * 0.40    # open window area, m^2
r_eq = np.sqrt(A_OPEN / np.pi)
L_EFF = 1.7 * r_eq      # thin opening: neck length is all end correction (flanged both sides)

f_helmholtz = C / (2 * np.pi) * np.sqrt(A_OPEN / (V_CABIN * L_EFF))
print(f"Helmholtz estimate: {f_helmholtz:.1f} Hz  (V = {V_CABIN} m^3, A = {A_OPEN:.2f} m^2)")

# ---------- measured spectrum
samples, fs = load_audio("data/nz_car_window_buffeting.m4a")

# 0.37 Hz resolution, averaged over overlapping segments to beat down broadband noise
f, psd = welch(samples, fs, nperseg=2 ** 17)
psd_db = 10 * np.log10(psd)

low = (f > 5) & (f < 30)
pk, props = find_peaks(psd_db[low], prominence=6)
best = pk[np.argmax(props["prominences"])]
f_tone, level = f[low][best], psd_db[low][best]
print(f"Measured tone: {f_tone:.1f} Hz, {props['prominences'].max():.0f} dB above the local floor")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
band = f <= 150
ax1.axvspan(40, 90, color="0.9", zorder=0)
ax1.text(65, psd_db[band].min() + 2, "broadband road/wind noise", ha="center", color="0.4")
ax1.plot(f[band], psd_db[band], lw=1)
ax1.plot(f_tone, level, "rv")
ax1.annotate(f"buffeting tone {f_tone:.1f} Hz", (f_tone, level), textcoords="offset points", xytext=(8, 6))
ax1.axvline(f_helmholtz, color="k", ls="--", lw=1)
ax1.annotate(f"Helmholtz estimate\n{f_helmholtz:.0f} Hz", (f_helmholtz, level - 12),
             textcoords="offset points", xytext=(6, 0))
ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("PSD [dB/Hz]")
ax1.set_title("Window buffeting: Welch power spectrum")
ax1.grid(alpha=0.3)

fs_t, ts, Sxx = spectrogram(samples, fs, nperseg=2 ** 15, noverlap=2 ** 14 + 2 ** 13)
keep = fs_t <= 150
ax2.pcolormesh(ts, fs_t[keep], 10 * np.log10(Sxx[keep] + 1e-20), shading="auto", cmap="magma")
ax2.axhline(f_tone, color="c", ls=":", lw=1)
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Frequency [Hz]")
ax2.set_title("Spectrogram: the tone holds steady for the whole recording")

fig.tight_layout()
fig.savefig("docs/car_buffeting.png", dpi=150, bbox_inches="tight")
plt.show()
