import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.io import fits
from astropy.time import Time
import config


def read_header(filepath):
    with fits.open(filepath) as hdul:
        return hdul[0].header.copy()

# its in the name what this does. calls on another function "read_header"
def file_discovery():
    pattern_fits = str(config.raw / "**" / "*.fits")
    files = glob.glob(pattern_fits, recursive=True)

    print(f"Found {len(files)} FITs files, continuing.")

    pop_time = Time(config.balloon_pop_time, format="isot", scale="utc")
    
    records = []
    skipped = []

    for filepath in files:
        try:
            hdr = read_header(filepath)

            raw_time = hdr.get(config.date_obs)
            if raw_time is None:
                print(f"[SKIP] {os.path.basename(filepath)}, no {config.date_obs} keyword found")
                continue
            obstime = Time(raw_time, format="isot", scale="utc")
            
            if obstime < pop_time:
                phase = "ascent"
            else:
                phase = "descent"

            # altitude = None # for later
            
            records.append({
                "filepath":filepath,
                "obstime":obstime,
                "phase":phase,
            })
        except Exception as e:
            print(f"[SKIP] {os.path.basename(filepath)}, {e}")
            skipped.append(filepath)
    if not records: 
        raise RuntimeError("No files read, check FITs headers and date_obs in config.py")
    
    # sorts by timestamp so as the index goes up, its in time order
    records.sort(key=lambda r: r["obstime"].unix)

    print(f"{len(records)} files loaded, {len(skipped)} skipped")
    print(f"Ascent frames: {sum(1 for r in records if r['phase'] == 'ascent')}")
    print(f"Descent frames: {sum(1 for r in records if r['phase'] == 'descent')}")
    return records

# reads 2d array from FITs, returns f32 np array like (rows, cols) = (y, x)
def read_frame(filepath):
    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float32)
    
    return data

# cuts down raw frame to region of interest as stated in config
def extract_roi(data):
    y0 = config.y_start
    y1 = config.y_start + config.y_height
    x0 = config.x_start
    x1 = config.x_start + config.x_width

    roi = data[y0:y1, x0:x1]
    return roi

# the juicy function (displays one frame, this is to test the positions)
def preview_frames(filepath=None):
    if filepath is None:
        files = glob.glob(str(config.raw / "**" / "*.fits"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No FITS files found in {config.raw}")
        filepath = files[0]
        print(f"No filepath given, previewing: {os.path.basename(filepath)}")
    data = read_frame(filepath)
    print(f"Frame shape: {data.shape} (rows={data.shape[0]}, cols={data.shape[1]})")

    fig, ax = plt.subplots(figsize=(14,8))

    ax.imshow(data, origin="upper", cmap="gray", aspect="auto")

    # this part draws ROI as red box
    rect = patches.Rectangle(
        (config.x_start, config.y_start),   # (col, row) of top-left corner
        config.x_width,                      # width in pixels  (x direction)
        config.y_height,                     # height in pixels (y direction)
        linewidth=2, edgecolor="red", facecolor="none",
        label="Extraction ROI"
    )
    ax.add_patch(rect)

    ax.set_title(
        f"{os.path.basename(filepath)}\n"
        f"ROI: x=[{config.x_start}:{config.x_start + config.x_width}]  "
        f"y=[{config.y_start}:{config.y_start + config.y_height}]",
        fontsize=10
    )
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig('preview_frame.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved preview_frame.png')

# pickle cache and processing logs
# saves results to a pickle cache, useful for not having consistently long runtimes
def save_cache(data_dict):
    config.cache.mkdir(parents=True, exist_ok=True)
    with open(config.pickle_cash, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Cache saved to {config.pickle_cash} ({len(data_dict)} entries)")

# loads cache when called
def load_cache():
    if not config.pickle_cash.exists():
        print("No cache found, restarting")
        return {}
    with open(config.pickle_cash, "rb") as f:
        data_dict = pickle.load(f)
    print(f"Cache loaded from {config.pickle_cash} ({len(data_dict)} entries)")
    return data_dict

# return set of basenames already fully processed
def load_processed_log():
    if not config.processed_log.exists():
        return set()
    return set(config.processed_log.read_text().splitlines())

# append a files basename to the processed log so it doesnt rerun
def mark_processed(filepath):
    config.cache.mkdir(parents=True, exist_ok=True)
    with open(config.processed_log, "a") as f:
        f.write(os.path.basename(filepath) + "\n")

