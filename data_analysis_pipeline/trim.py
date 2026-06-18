# grab spectra from the ascent of the launch, the peak of the flight, and maybe in descent?

import os
import shutil
import numpy as np
import glob
from astropy.io import fits
from pathlib import Path
import config

# thresholds
sat_thresh = 45000 # adu value considered saturated
max_sat_frac = 0.40 # reject if more than x percent are at or near max
min_median = 50 # reject frames that are dark or empty, given by x value in adu

trim_directory = config.base / "inital_trim"

def load_frame(filepath):
    with fits.open(filepath) as hdul:
        return hdul[0].data.astype(np.float32)
    
# returns true if frame should be rejected, false if it passes, add stuff as needed
def is_bad(data):
    sat_fraction = np.mean(data >= sat_thresh)
    if sat_fraction > max_sat_frac:
        return True, f"saturated ({sat_fraction * 100:.1f}% >= {sat_thresh} ADU)"

    if np.median(data) < min_median:
        return True, f"underexposed (median = {np.median(data):.1f} ADU)"

    return False, None

# wipe and recreate stuff inside folder each run so no duplicates
def main():
    if trim_directory.exists():
        shutil.rmtree(trim_directory)
        print(f"Cleared existing {trim_directory}")
    trim_directory.mkdir(parents=True)
    print(f"Writing good frames to: {trim_directory}\n")

    # discover all FITS files in raw_data
    pattern = str(config.raw / "**" / "*.fits")
    all_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(all_files)} total FITS files\n")

    n_kept     = 0
    n_rejected = 0

    for filepath in all_files:
        filename = os.path.basename(filepath)
        try:
            data = load_frame(filepath)
            bad, reason = is_bad(data)

            if bad:
                print(f"Rejecting {filename} because {reason}")
                n_rejected += 1
            else:
                # copy good frame into initial_trim
                shutil.copy2(filepath, trim_directory / filename)
                n_kept += 1

        except Exception as e:
            print(f"  ERROR   {filename}  —  {e}")
            n_rejected += 1

    print(f"\n=== Trim complete ===")
    print(f"Kept: {n_kept}")
    print(f"Rejected: {n_rejected}")
    print(f"Output: {trim_directory}")
