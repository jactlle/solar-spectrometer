import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import config

# input uncertainty later

# three gaussian
def _multi_gaussian(x, baseline, amp1, cen1, sig1, amp2, cen2, sig2, amp3, cen3, sig3):
    g1 = amp1 * np.exp(-0.5 * ((x - cen1) / sig1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - cen2) / sig2) ** 2)
    g3 = amp3 * np.exp(-0.5 * ((x - cen3) / sig3) ** 2)
    return baseline + g1 + g2 + g3

# fit the three gaussians to the thing
def fit_o2_center(spectrum, window=None):
    if window is None:
        window = config.searchline

    # use a wider window
    lo = max(0, config.o2_approx_px - window * 3)
    hi = min(len(spectrum), config.o2_approx_px + window * 3)
    x  = np.arange(lo, hi)
    y  = spectrum[lo:hi]

    y = np.where(np.isnan(y), np.nanmedian(y), y)

    baseline_est = np.percentile(y, 90)  # top of the band, not the bottom

    # initial guesses for 3 sub-features using config approximate positions
    # amplitudes are negative (absorption dips)
    amp_est = np.min(y) - baseline_est
    p0 = [baseline_est, amp_est, config.o2_sub1_approx_px, window / 3.0, amp_est, config.o2_sub2_approx_px, window / 3.0, amp_est, config.o2_sub3_approx_px, window / 3.0,]

    # amplitudes must be negative, centers within their local windows,
    # sigmas must be positive and reasonable
    w = window
    bounds_lo = [-np.inf,
                 -np.inf, config.o2_sub1_approx_px - w, 0.5,
                 -np.inf, config.o2_sub2_approx_px - w, 0.5,
                 -np.inf, config.o2_sub3_approx_px - w, 0.5]
    bounds_hi = [np.inf,
                 0, config.o2_sub1_approx_px + w, w * 2,
                 0, config.o2_sub2_approx_px + w, w * 2,
                 0, config.o2_sub3_approx_px + w, w * 2]

    try:
        popt, pcov = curve_fit(_multi_gaussian, x, y, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=10000,)

        # unpack fitted parameters
        _, amp1, cen1, _, amp2, cen2, _, amp3, cen3, _ = popt

        # amplitude-weighted mean center, stable 
        abs_amps = [abs(amp1), abs(amp2), abs(amp3)]
        centers  = [cen1, cen2, cen3]
        weighted_center = np.average(centers, weights=abs_amps)

        # propagate uncertainty from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        # uncertainty on weighted center, simplified quadrature of center errors
        center_unc = np.sqrt(sum((a * e)**2 for a, e in zip(abs_amps, perr[2::3]))) / sum(abs_amps)

        print(f"O2 sub-centers: {cen1:.1f}, {cen2:.1f}, {cen3:.1f} px, weighted mean: {weighted_center:.2f} px")
        return weighted_center, center_unc
    
    except RuntimeError:
        print(f"Multi-Gaussian O2 fit failed, falling back to single Gaussian")
        return fit_line_center(spectrum, config.o2_approx_px)

# standard gaussian. keep in mind need to fit multiple gaussians to the o2
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
        popt, pcov = curve_fit(_gaussian, x, y, p0=[amplitude_est, center_est, sigma_est, baseline_est], maxfev=5000,) # pcov = covariance, popt = parameter optimum
        fitted_center = popt[1]
        center_uncertainty = np.sqrt(pcov[1, 1]) # 1 sigma error on center
        # fitted center must stay within the search window
        if not (lo < fitted_center < hi):
            print(f"Gaussian center {fitted_center:.1f} drifted outside search window [{lo}, {hi}] near px {approx_px}")
            return None

        return fitted_center, center_uncertainty

    except RuntimeError:
        print(f"Gaussian fit failed near px {approx_px}")
        return None

# wavelength mapping turning two pixel positions into formula
def build_wavelength_solution(spectrum):
    px_halpha, err_halpha = fit_line_center(spectrum, config.halpha_approx_px) # value from config
    px_o2, err_o2 = fit_line_center(spectrum, config.o2_approx_px)

    if px_halpha is None or px_o2 is None:
        print("Put in a line fit that doesnt return None bro")
        return None, None
    
    # dispersion from two anchor points using delta_wavelength / delta_pixel
    dispersion = (config.oxygen_2 - config.halpha) / (px_o2 - px_halpha) # in A/px
    zero_point = config.halpha - dispersion*px_halpha 

    px_gap = px_o2 - px_halpha
    sigma_disp = abs(dispersion) * np.sqrt((err_halpha / px_gap)**2 + (err_o2/ px_gap)**2) # frames where its shaky will have high sigma disp

    # should be .237 or so A/px
    if not (0.08 < abs(dispersion) < 0.40):
        print(f"Dispersion {dispersion:.4f} is outside range")
    print(f"H-alpha center: {px_halpha:.2f} px")
    print(f"O2 center: {px_o2:.2f} px")
    print(f"Dispersion: {dispersion:.4f} A/px")
    return dispersion, zero_point

# convert pixel array to wavelength in angstrom
def apply_wavelength_solution(pixels, dispersion, zero_point):
    return dispersion*np.asarray(pixels) + zero_point # from wavelengthFromPixel() in the notebook

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

    filepath = "/home/jacobt/Real Atmospheric Data/inital_trim/20260612_160531_exp104563_gain380.fits"

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
        print("Wavelength solution failed, check config.py approx pixel values")

