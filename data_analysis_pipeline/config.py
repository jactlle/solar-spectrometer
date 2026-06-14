import os
from pathlib import Path

# directories
base = Path(os.path.expanduser("~/Real Atmospheric Data"))
raw = base / "raw_data"
output = base / "calibrated_out"
cache = base / "cache"

# cache and log
pickle_cash = cache / "SpectraCache.pkl" # stores extracted spectra and pixel shifts to speed up re-run's
processed_log = cache / "processed_files.txt" # frames listed in here are skipped on re-run's

# trace. dispersion axis is VERTICAL (y = wavelength), spatial axis is HORIZONTAL (x)
x_start = 50 # first column to include
x_width = 780 # how many columns to sum: tuneable
y_start = 0 # first usable row
y_height = 2100 # crop before black bar on bottom: tuneable

# flight timing stuff
balloon_pop_time = "2025-06-14T16:50:56"
date_obs = "DATE-OBS"

# reference lines wavelength (in angstroms)
halpha = 6562.8
oxygen_2 = 6867.0 # i think

searchline = 80 # search window for fitting line on reference frame

# equivalent width: all tuneable
halpha_halfwidth = 8.0 # measure 8 angstroms to either side of centered line 
halpha_cont_offset = 20.0 # gap between lines edge and the start of the continuum
halpha_cont_width = 15.0 # width of each continuum sample region

o2_halfwidth = 35.0 # same as halpha but wider
o2_cont_offset = 45.0 # thus the continuum starts further out 
o2_cont_width = 20.0