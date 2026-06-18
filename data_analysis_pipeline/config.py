import os
from datetime import datetime
from pathlib import Path

# directories
base = Path(os.path.expanduser("~/Real Atmospheric Data"))
raw = base / "inital_trim"
output = base / "calibrated_out"
cache = base / "cache"

# cache and log
pickle_cash = cache / "SpectraCache.pkl" # stores extracted spectra and pixel shifts to speed up re-run's
processed_log = cache / "processed_files.txt" # frames listed in here are skipped on re-run's

# trace. dispersion axis is VERTICAL (y = wavelength), spatial axis is HORIZONTAL (x)
x_start = 300 # first usable row
x_width = 3000 # how many columns to sum
y_start = 750 # first column to include 
y_height = 60 # crop before black bar on bottom

# flight timing stuff
# balloon_pop_time = datetime(2026, 6, 12, 16, 50, 56)
balloon_pop_time = "2026-06-12T16:50:56" 
date_obs = "DATE-OBS"

# reference lines wavelength (in angstroms)
halpha = 6562.8
oxygen_2 = 6867 # i think
halpha_approx_px = 1100
o2_approx_px = 2110
o2_sub1_approx_px = 2100   # left sub-feature
o2_sub2_approx_px = 2110  # central deep dip 
o2_sub3_approx_px = 2150   # right sub-feature

searchline = 80 # search window for fitting line on reference frame

# equivalent width stuff
halpha_halfwidth = 15.0 # measure set amount of angstroms to either side of centered line 
halpha_cont_offset = 20.0 # gap between lines edge and the start of the continuum
halpha_cont_width = 15.0 # width of each continuum sample region

o2_left_only = True
o2_halfwidth = 8.0 # same idea
o2_cont_offset = 45.0 # thus the continuum starts further out 
o2_cont_width = 15.0
o2_drop_threshold = 0.995
o2_band_max = 6940
o2_ew_max = 25

min_snr = 60
max_cont_slope = 30
min_line_depth = 0.05