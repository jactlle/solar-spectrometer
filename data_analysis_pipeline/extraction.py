import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

import config
from utils import read_frame, extract_roi

# main function here where it uses other functions to return the spectrum
def extract_1d(filepath):
    data = read_frame(filepath)
    roi = extract_roi(data)
    roi = sigma_clip_frame(roi)
    spectrum = collapse(roi)
    return spectrum

# filters cosmic rays / hot pixels before collapse, goes column by column
def sigma_clip_frame(roi, sigma=3.0, iterations=2):
    masked = ma.array(roi, mask=np.zeros_like(roi, dtype=bool)) # marks all pixels as good initially

    for _ in range(iterations):
        col_median = ma.median(masked, axis=0, keepdims=True) # For each column, computes median and standard dev, ignoring masked pixels (ma), 
        col_std = ma.std(masked, axis=0,keepdims=True)

        outliers = np.abs(masked - col_median) > sigma*col_std # essentially asking if a pixel is more than 3 std dev's away from the median, if yes, then outlier
        masked = ma.array(roi, mask=masked.mask | outliers.filled(False)) # 

    n_masked = masked.mask.sum()
    if n_masked > 0:
        print(f"Sigma clipped (masked) {n_masked} pixels")
    return masked

# the median along x produces 1d spectrum
def collapse(roi):
    spectrum = ma.median(roi, axis=0)

    # convert from masked array to plain array, and if all pixels were masked in a row, then its filled with nan and treated as bad data
    return np.where(spectrum.mask if hasattr(spectrum, 'mask') else np.zeros(len(spectrum), dtype=bool), np.nan,
                    spectrum.data if hasattr(spectrum, 'data') else np.array(spectrum))

# plotting function
def preview_spectrum(filepath=None, spectrum=None):
    if spectrum is None:
        if filepath is None:
            raise ValueError("Pass either filepath or spectrum.")
        spectrum = extract_1d(filepath)

    pixels = np.arange(len(spectrum))

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(pixels, spectrum, color="steelblue", linewidth=0.8, label="1D Spectrum")

    # marking approximate locations of halpha and o2
    halpha_approx = 1100 # both these values came from guessing until it worked
    o2_approx = 2115

    ax.axvline(halpha_approx, color="red", linestyle="--", alpha=0.6, label=f"H-alpha approximation line")
    ax.axvline(o2_approx, color="orange", linestyle="--", alpha=0.6, label=f"O2 approximation line")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Median flux (ADU)")
    ax.set_title(f"Extracted 1D Spectrum")
    ax.legend()
    plt.tight_layout()
    plt.savefig("preview_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved preview_spectrum.png")


spectrum = extract_1d("/home/jacobt/Real Atmospheric Data/raw_data/20260612_171513_exp70390_gain380.fits")
preview_spectrum(spectrum=spectrum)
