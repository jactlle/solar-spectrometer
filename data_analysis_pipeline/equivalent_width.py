import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from astropy import units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.analysis import equivalent_width
from specutils.fitting import fit_continuum
from astropy.modeling.models import Chebyshev1D

import config


# calculating edges of continuum regions
def _get_continuum_windows(wavelengths, line_center, cont_offset, cont_width):
    left_lo  = line_center - cont_offset - cont_width # cont_offset is gap between line edge and continuum start
    left_hi  = line_center - cont_offset # cont_width is how wide each continuum sample window is 
    right_lo = line_center + cont_offset
    right_hi = line_center + cont_offset + cont_width

    # true wherever the wavelength array falls inside each window
    left_mask  = (wavelengths >= left_lo)  & (wavelengths <= left_hi)
    right_mask = (wavelengths >= right_lo) & (wavelengths <= right_hi)

    return left_mask, right_mask

# for o2, using specutils
def _fit_continuum_spec(wavelengths, flux, left_mask):
    spec = Spectrum1D(spectral_axis=wavelengths * u.AA, flux=flux * u.adu)

    # only left
    left_lo = float(wavelengths[left_mask][0])
    left_hi = float(wavelengths[left_mask][-1])
    window  = SpectralRegion(left_lo * u.AA, left_hi * u.AA)

    cont_model = fit_continuum(spec, model=Chebyshev1D(1), window=window)

    # evaluate over the full wavelength array
    continuum = cont_model(wavelengths * u.AA).value
    slope     = (continuum[-1] - continuum[0]) / (wavelengths[-1] - wavelengths[0])

    return continuum, slope

# fits continuum model
def _fit_continuum(wavelengths, flux, left_mask, right_mask, left_only=False):
    if left_only:
        cont_wave = wavelengths[left_mask]
        cont_flux = flux[left_mask]
    else:
        cont_wave = np.concatenate([wavelengths[left_mask], wavelengths[right_mask]])
        cont_flux = np.concatenate([flux[left_mask], flux[right_mask]])

    # need at least 4 points to fit a meaningful line
    if len(cont_wave) < 4:
        return None, None, None

    # fit a straight line through the continuum samples
    # this models the local baseline shape across the line region
    slope, intercept, _, _, _ = linregress(cont_wave, cont_flux)

    # evaluate the continuum model at every wavelength point in the full spectrum
    continuum = slope * wavelengths + intercept
    return continuum, slope, intercept

# computes signal to noise ratio
def _compute_snr(flux, left_mask, right_mask=None, left_only=False):
    if left_only:
        cont_flux = flux[left_mask]
    else:
        cont_flux = np.concatenate([flux[left_mask], flux[right_mask]])
    if len(cont_flux) < 4:
        return np.nan
    return np.mean(cont_flux)/np.std(cont_flux)

# doing the actual ew calculation (may need to tweak a bit) and for halpha for now 
def measure_ew(wavelengths, flux, line_center, half_width, cont_offset, cont_width, line_name="line", left_only=False):
    left_mask, right_mask = _get_continuum_windows(wavelengths, line_center, cont_offset, cont_width)

    if left_mask.sum() < 2 or right_mask.sum() < 2:
        print(f"[{line_name}]: continuum window has too few pixels")
        return None

    continuum, slope, _ = _fit_continuum(wavelengths, flux, left_mask, right_mask, left_only=left_only)
    if continuum is None:
        print(f"[{line_name}]: continuum fit failed")
        return None

    # normalize
    normalized = flux / continuum

    # integration window
    line_mask = (wavelengths >= line_center - half_width) & \
                (wavelengths <= line_center + half_width)

    if line_mask.sum() < 3:
        print(f"[{line_name}]: line window has too few pixels")
        return None

    wave_window = wavelengths[line_mask]
    norm_window = normalized[line_mask]
    # use multiple line centers and for now, add together 
    # EW = integral of (1 - normalized_flux) dlambda


    ew = np.trapezoid(1.0 - norm_window, wave_window)

    snr = _compute_snr(flux, left_mask, right_mask, left_only=left_only)
    line_depth = 1.0 - np.min(norm_window)

    n_pix = line_mask.sum()
    delta_lambda = wave_window[-1] - wave_window[0]
    ew_uncertainty = delta_lambda / (snr * np.sqrt(n_pix)) if np.isfinite(snr) else np.nan

    flagged = (snr < config.min_snr or abs(slope) > config.max_cont_slope or line_depth < config.min_line_depth)

    if flagged:
        print(f"flagged [{line_name}]: SNR={snr:.1f}, slope={slope:.5f}, depth={line_depth:.3f}")

    return {
        "ew": ew,
        "ew_unc": ew_uncertainty,
        "snr": snr,
        "cont_slope": slope,
        "line_depth": line_depth,
        "flagged": flagged,
    }

# for o2 
def measure_ew_o2(wavelengths, flux):
    left_mask, _ = _get_continuum_windows(wavelengths, config.oxygen_2, config.o2_cont_offset, config.o2_cont_width)

    if left_mask.sum() < 4:
        print("[O2]: continuum window has too few pixels")
        return None

    continuum, slope= _fit_continuum_spec(wavelengths, flux, left_mask)
    if continuum is None:
        print("[O2]: continuum fit failed")
        return None

    snr = _compute_snr(flux, left_mask, left_only=True)

    normalized = flux / continuum

    # dynamic edge detection
    search_start = config.oxygen_2 - config.o2_cont_offset
    search_mask  = (wavelengths >= search_start) & (wavelengths <= config.o2_band_max)

    wave_search = wavelengths[search_mask]
    norm_search = normalized[search_mask]

    threshold = config.o2_drop_threshold    # pulled from config 
    below = np.where(norm_search < threshold)[0]

    if len(below) == 0:
        print("[O2]: no absorption detected above threshold")
        return None

    left_idx  = below[0]
    right_idx = below[-1]

    if right_idx - left_idx < 3:
        print("[O2]: absorption region too narrow")
        return None

    band_left  = float(wave_search[left_idx])
    band_right = float(wave_search[right_idx])

    # use specutils
    norm_spec = Spectrum1D(spectral_axis=wavelengths * u.AA, flux=normalized * u.dimensionless_unscaled)
    ew_region = SpectralRegion(band_left * u.AA, band_right * u.AA)
    ew_result = equivalent_width(norm_spec, regions=ew_region)
    ew        = float(ew_result.value)

    line_depth = 1.0 - float(np.min(norm_search[left_idx : right_idx + 1]))

    n_pix = right_idx - left_idx + 1
    delta_lambda = band_right - band_left
    ew_uncertainty = delta_lambda / (snr * np.sqrt(n_pix)) if np.isfinite(snr) else np.nan

    flagged = (snr < config.min_snr or ew < 0 or ew > config.o2_ew_max or line_depth < config.min_line_depth) # no negative ew

    return {
        "ew":         ew,
        "ew_unc":     ew_uncertainty,
        "snr":        snr,
        "cont_slope": slope,
        "line_depth": line_depth,
        "flagged":    flagged,
        "band_left":  band_left,    
        "band_right": band_right,
    }

# measuring ew for both ha and o2, this is why I kept it general in the other functions so I can assign it here
def measure_all(wavelengths, flux):
    ha = measure_ew(wavelengths, flux, line_center=config.halpha, half_width=config.halpha_halfwidth, cont_offset=config.halpha_cont_offset, cont_width=config.halpha_cont_width, line_name="H-alpha")
    o2 = measure_ew_o2(wavelengths, flux)
    return ha, o2

# plot preview (for testing)
def preview_ew(wavelengths, flux, filepath=None):
    ha, o2 = measure_all(wavelengths, flux)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # H-alpha panel
    ax = axes[0]
    if ha is None:
        ax.set_title("H-alpha: fit failed")
    else:
        left_mask, right_mask = _get_continuum_windows(
            wavelengths, config.halpha,
            config.halpha_cont_offset, config.halpha_cont_width
        )
        continuum, _, _ = _fit_continuum(wavelengths, flux, left_mask, right_mask, left_only=False)

        plot_lo   = config.halpha - config.halpha_cont_offset - config.halpha_cont_width - 5
        plot_hi   = config.halpha + config.halpha_cont_offset + config.halpha_cont_width + 5
        plot_mask = (wavelengths >= plot_lo) & (wavelengths <= plot_hi)

        ax.plot(wavelengths[plot_mask], flux[plot_mask], color="steelblue", linewidth=0.9, label="Flux")
        ax.plot(wavelengths[plot_mask], continuum[plot_mask], color="green", linestyle="--", linewidth=1.2, label="Continuum fit")

        line_mask = (
            (wavelengths >= config.halpha - config.halpha_halfwidth) &
            (wavelengths <= config.halpha + config.halpha_halfwidth)
        )
        ax.fill_between(
            wavelengths[line_mask], flux[line_mask], continuum[line_mask],
            alpha=0.3, color="red", label=f"EW = {ha['ew']:.2f} Å"
        )

        # shade the left and right continuum sample windows
        for mask in [left_mask, right_mask]:
            plot_cont_mask = mask & (wavelengths >= plot_lo) & (wavelengths <= plot_hi)
            if plot_cont_mask.sum() > 0:
                ax.axvspan(wavelengths[plot_cont_mask][0], wavelengths[plot_cont_mask][-1], alpha=0.1, color="green")

        flag_str = " flagged" if ha["flagged"] else ""
        ax.set_title(f"H-alpha{flag_str}\nEW={ha['ew']:.2f} Å  SNR={ha['snr']:.1f}  depth={ha['line_depth']:.3f}")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux (ADU)")
        ax.legend(fontsize=8)

    # O2 part
    ax = axes[1]
    if o2 is None:
        ax.set_title("O2: fit failed")
    else:
        left_mask, _ = _get_continuum_windows(
            wavelengths, config.oxygen_2,
            config.o2_cont_offset, config.o2_cont_width
        )
        continuum, _, _ = _fit_continuum(wavelengths, flux, left_mask, right_mask=None, left_only=True)

        plot_lo   = config.oxygen_2 - config.o2_cont_offset - config.o2_cont_width - 5
        plot_hi   = o2["band_right"] + 10  # extend plot to show the full detected band
        plot_mask = (wavelengths >= plot_lo) & (wavelengths <= plot_hi)

        # shade the detected O2 absorption band
        band_mask = (wavelengths >= o2["band_left"]) & (wavelengths <= o2["band_right"])

        ax.plot(wavelengths[plot_mask], flux[plot_mask], color="steelblue", linewidth=0.9, label="Flux")
        ax.plot(wavelengths[plot_mask], continuum[plot_mask], color="green", linestyle="--", linewidth=1.2, label="Continuum fit")
        ax.fill_between(
            wavelengths[band_mask], flux[band_mask], continuum[band_mask],
            alpha=0.3, color="orange",
            label=f"EW = {o2['ew']:.2f} Å\n{o2['band_left']:.1f}–{o2['band_right']:.1f} Å"
        )

        # shade the left continuum sample window
        plot_cont_mask = left_mask & (wavelengths >= plot_lo) & (wavelengths <= plot_hi)
        if plot_cont_mask.sum() > 0:
            ax.axvspan(wavelengths[plot_cont_mask][0], wavelengths[plot_cont_mask][-1], alpha=0.1, color="green")

        flag_str = " flagged" if o2["flagged"] else ""
        ax.set_title(f"O2{flag_str}\nEW={o2['ew']:.2f} Å  SNR={o2['snr']:.1f}  depth={o2['line_depth']:.3f}")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux (ADU)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = "preview_ew.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    

# test block
if __name__ == "__main__":  
    from extraction import extract_1d
    from wavelength import build_wavelength_solution, apply_wavelength_solution

    filepath = "/home/jacobt/Real Atmospheric Data/inital_trim/20260612_160531_exp104563_gain380.fits"

    print("Extracting spectrum...")
    spectrum = extract_1d(filepath)

    print("Building wavelength solution...")
    dispersion, zero_point = build_wavelength_solution(spectrum)

    if dispersion is not None:
        pixels      = np.arange(len(spectrum))
        wavelengths = apply_wavelength_solution(pixels, dispersion, zero_point)

        print("Measuring equivalent widths...")
        ha, o2 = measure_all(wavelengths, spectrum)

        if ha:
            print(f"H-alpha EW: {ha['ew']:.3f} A  (SNR={ha['snr']:.1f}, depth={ha['line_depth']:.3f}, flagged={ha['flagged']})")
        if o2:
            print(f"O2 B-band EW: {o2['ew']:.3f} A  (SNR={o2['snr']:.1f}, depth={o2['line_depth']:.3f}, flagged={o2['flagged']})")

        preview_ew(wavelengths, spectrum)
    else:
        print("Wavelength solution failed, fix wavelength.py")