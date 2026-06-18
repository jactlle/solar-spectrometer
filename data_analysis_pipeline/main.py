import os 
import csv
import traceback
import numpy as np 

import config
from utils import (file_discovery, load_cache, save_cache, load_processed_log, mark_processed, read_frame)
from extraction import extract_1d
from wavelength import build_wavelength_solution, apply_wavelength_solution
from equivalent_width import measure_all

sat_thresh = 45000
max_sat_frac = 0.40
min_median = 50

#quick screening for frames coming through
def screen_frame(filepath):
    data = read_frame(filepath)

    sat_fraction = np.mean(data >= sat_thresh)
    if sat_fraction > max_sat_frac:
        return True, f"saturated ({sat_fraction * 100:.1f}% of pixels >= {sat_thresh})"

    # dark or empty frames
    if np.median(data) < min_median:
        return True, f"underexposed (median = {np.median(data):.1f} ADU)"

    return False, None

# single frame pipeline, extracts spectrum, builds wavelength solution, measures ew + uncertainties
def process_frame(record):
    filepath = record["filepath"]
    obstime = record["obstime"] # recording time
    phase = record["phase"]  # recording ascent or descent

    # sigma clip and collapse into 1d spectrum
    spectrum = extract_1d(filepath)

    # fit gaussians and get the good stuff
    result = build_wavelength_solution(spectrum)

    # handle any errors up to this point to pinpoint 
    if result is None or result[0] is None:
        raise ValueError("Wavelength solution failed check your approximates")
    if len(result) == 3:
        dispersion, zero_point, sigma_disp = result
    else:
        dispersion, zero_point = result
        sigma_disp = None

    pixels = np.arange(len(spectrum))
    wavelengths = apply_wavelength_solution(pixels, dispersion, zero_point)

    # continuum fit and equivalent width calc for both
    ha, o2 = measure_all(wavelengths, spectrum)
    
    # Put results into prefixed CSV columns
    def unpack(result, prefix):
        if result is None:
            return {
                f"{prefix}_ew": None,
                f"{prefix}_ew_unc": None,
                f"{prefix}_snr": None,
                f"{prefix}_slope": None,
                f"{prefix}_depth": None,
                f"{prefix}_flagged": None
            }
        return {
            f"{prefix}_ew": round(result["ew"], 4),
            f"{prefix}_ew_unc": round(result.get("ew_unc", float("nan")), 4),
            f"{prefix}_snr": round(result["snr"], 2),
            f"{prefix}_slope": round(result["cont_slope"], 6),
            f"{prefix}_depth": round(result["line_depth"], 4),
            f"{prefix}_flagged": result["flagged"],
        }

    return {
        "filename": os.path.basename(filepath),
        "obstime": obstime.isot, # ISO 8601 so its easy to load with pandas later
        "phase": phase,
        "dispersion": round(dispersion,  6), # A/px for checking
        "zero_point": round(zero_point,  4),
        "sigma_disp": round(sigma_disp,  6) if sigma_disp is not None else None,
        **unpack(ha, "ha"),
        **unpack(o2, "o2"),
    }

# general layout for csv
csv_fields = [
    "filename", "obstime", "phase",
    "dispersion", "zero_point", "sigma_disp",
    "ha_ew",  "ha_ew_unc",  "ha_snr",  "ha_slope",  "ha_depth",  "ha_flagged",
    "o2_ew",  "o2_ew_unc",  "o2_snr",  "o2_slope",  "o2_depth",  "o2_flagged",
]

output_csv = config.output / "results.csv"

# main loop
def main():
    print("Here we go")
    
    # checking for directory existence
    config.output.mkdir(parents=True, exist_ok=True)
    config.cache.mkdir(parents=True, exist_ok=True)

    # discover all FITS files sorted by DATE-OBS, tagged ascent/descent
    records = file_discovery()

    # pickle cache stores completed result dicts for fast re-access (e.g. plotting)
    cache = load_cache()

    # processed log is a plain text list of completed filenames, anything listed here is skipped instantly so you can safely kill and restart
    processed = load_processed_log()

    # open CSV, no overwriting
    csv_exists = output_csv.exists()
    csv_file = open(output_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        writer.writeheader()  # only write header on a completely fresh run

    n_done = 0
    n_skipped = 0
    n_screened = 0
    n_failed = 0

    for i, record in enumerate(records):
        filename = os.path.basename(record["filepath"])
        prefix = f"[{i+1}/{len(records)}]  {filename}  ({record['phase']})"

        # for if thing was already completed, skip
        if filename in processed:
            n_skipped += 1
            continue

        print(prefix, end="  ")

        # screen check
        bad, reason = screen_frame(record["filepath"])
        if bad:
            print(f"SCREENED — {reason}")
            mark_processed(record["filepath"])  # don't retry screened frames next run
            n_screened += 1
            continue

        # now run pipeline
        try:
            row = process_frame(record)

            # write and flush immediately so interruption doesnt fuck everything up
            writer.writerow(row)
            csv_file.flush()

            # update cache and processed log
            cache[filename] = row
            mark_processed(record["filepath"])

            # flag any bad quality shi
            flag_str = ""
            if row["ha_flagged"]: flag_str += "  [HA flagged]"
            if row["o2_flagged"]: flag_str += "  [O2 flagged]"
            print(f"HA={row['ha_ew']} Å   O2={row['o2_ew']} Å{flag_str}")
            n_done += 1

        except Exception as e:
            # log error and keep going, intentionally NOT marking as processed so failed frames automatically retry on the next run
            print(f"Failed: {e}")
            traceback.print_exc()
            n_failed += 1

    csv_file.close()
    save_cache(cache)  # save updated pickle cache once at the end

    print(f"\nResults")
    print(f" Processed this run : {n_done}")
    print(f" Skipped (cached) : {n_skipped}")
    print(f" Screened (bad) : {n_screened}")
    print(f" Failed : {n_failed}")
    print(f" Results CSV : {output_csv}")


if __name__ == "__main__":
    main()