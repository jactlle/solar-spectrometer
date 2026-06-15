import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import config

# standard gaussian
def _gaussian(x, amplitude, center, sigma, baseline):
    return baseline + amplitude*np.exp(-0.5*((x-center)/sigma)**2) # basic formula

# fits the gaussian to absorption dip to find exact pixel center
def fit_line_center(spectrum, approx_px, window=None):
    if window is None:
        window = config.searchline
    
    lo = max(0, int(approx_px)-window)
    hi = min(len(spectrum), int(approx_px)+window)
    x = np.arange(lo, hi)
    y = spectrum[lo:hi]

    y = np.where(np.isnan(y), np.nanmedian(y), y)

    baseline_est = np.median(y)
    amplitude_est = np.min(y)-baseline_est
    center_est = x[np.argmin(y)]
    sigma_est = window/3.0

    try:
        popt, _ = curve_fit(_gaussian, x, y, p0=[amplitude_est, center_est, sigma_est, baseline_est], maxfev=5000,)
        fitted_center = popt[1]
        # fitted center must stay within the search window
        if not (lo < fitted_center < hi):
            print(f"Gaussian center {fitted_center:.1f} drifted outside search window [{lo}, {hi}] near px {approx_px}")
            return None

        return fitted_center

    except RuntimeError:
        print(f"Gaussian fit failed near px {approx_px}")
        return None

# wavelength mapping turning two pixel positions into formula
def build_wavelength_solution(spectrum):
    px_halpha = fit_line_center(spectrum, config.halpha_approx_px)
    px_o2 = fit_line_center(spectrum, config.o2_approx_px) # from config

    if px_halpha is None or px_o2 is None:
        print("Put in a line fit that doesnt return None bro")
        return None, None
    
    # dispersion from two anchor points using delta_wavelength / delta_pixel
    dispersion = (config.oxygen_2 - config.halpha) / (px_o2 - px_halpha) # in A/px
    zero_point = config.halpha - dispersion*px_halpha 

    # should be .14 or so A/px
    if not (0.08 < abs(dispersion) < 0.25):
        print(f"Dispersion {dispersion:.4f} is outside range")
    print(f"H-alpha center: {px_halpha:.2f} px")
    print(f"O2 center: {px_o2:.2f} px")
    print(f"Dispersion: {dispersion:.4f} A/px")
    return dispersion, zero_point

# convert pixel array to wavelength in angstrom
def apply_wavelength_solution(pixels, dispersion, zero_point):
    return dispersion*np.asarray(pixels) + zero_point # from wavelengthFromPixel()

# plot it
def preview_wavelength(spectrum, dispersion, zero_point, filepath=None):
    pixels = np.arange(len(spectrum))
    wavelengths = apply_wavelength_solution(pixels, dispersion, zero_point)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(wavelengths, spectrum, color="steelblue", linewidth=0.8, label="1D Spectrum")

    ax.axvline(config.halpha,   color="red",    linestyle="--", alpha=0.7, label=f"H-alpha ({config.halpha} Å)")
    ax.axvline(config.oxygen_2, color="orange", linestyle="--", alpha=0.7, label=f"O2 B-band ({config.oxygen_2} Å)")

    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Median flux (ADU)")
    ax.set_title(f"Wavelength-calibrated spectrum\n, dispersion = {dispersion:.4f} Å/px, zero point = {zero_point:.2f} Å")
    ax.legend()
    plt.tight_layout()
    plt.savefig("preview_wavelength.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved preview_wavelength.png")

# test block
if __name__ == "__main__":
    from extraction import extract_1d

    filepath = "/home/jacobt/Real Atmospheric Data/raw_data/20260612_171513_exp70390_gain380.fits"

    print("Extracting spectrum...")
    spectrum = extract_1d(filepath)

    print("Building wavelength solution...")
    dispersion, zero_point = build_wavelength_solution(spectrum)

    if dispersion is not None:
        pixels      = np.arange(len(spectrum))
        wavelengths = apply_wavelength_solution(pixels, dispersion, zero_point)
        preview_wavelength(spectrum, dispersion, zero_point)
        print(f"\nWavelength axis: {wavelengths[0]:.1f} — {wavelengths[-1]:.1f} Å")
    else:
        print("Wavelength solution failed — check config.py approx pixel values")