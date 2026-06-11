#!/usr/bin/env python

import os
import sys
import time # time.sleep() for settling loop
import zwoasi as asi # camera wrapper
import json
from datetime import datetime, timezone
import subprocess
import numpy as np
from astropy.io import fits
from gpiozero import CPUTemperature

# variable definitions
cadence = 15 # in seconds
target_mean = 25000 # pixel target for spectra? 
tolerance = 1000 # within 1000 of target? ok
exposure = 90000 # starting guess (90 ms)
exp_min = 1000 # minimum allowed (1 ms)
exp_max = 10000000 # maximum allowed (10 s) may be lowered
gain = 380 # for low light spectra, can be changed and mostly likely will be

# TO DO : Figure out what happens to the time when you disconnect wifi, power cycle, and reconnect wifi and figure out a solution HOW THE FUCK TO TRACK TIME
# TO DO : find a function that relates the change of altitude with the brightness, and implement it into influencing what the target mean will be?
# need to add part that allows an image to reach a mean target count of around 3000 or so so spectra taken at high altitude is not at exposure of 100s
# So put something where if the the exposure is above a certain time, use that time / 100 and see if it produces a good enough spectrum 
# Done I think, need to test 

# gathering file names - Edit here 
env_filename = os.path.expanduser('~/zwo/libASICamera2.so') # Need to figure out how to download ZWO_ASI_LIB 
save_directory = os.path.expanduser('~/spectrometer/captures')
status_filename = 'flight_status.json' # health file
# if not env_filename:
#     print('ZWO_ASI_LIB not set, exiting')
#     sys.exit(1)
asi.init(env_filename) # need this part to pass 

def utc_now():
    return datetime.now(timezone.utc)

def utc_file_timestamp():
    """
    UTC timestamp for filenames
    """
    return utc_now().strftime('%Y%m%d_%H%M%S')

def utc_iso_timestamp():
    """
    ISO-8601 UTC timestamp for metadata/logging
    example: 2026-06-10T02:03:45.123Z
    """
    return utc_now().isoformat(timespec='milliseconds').replace('+00:00', 'Z')

def clock_synchronized():
    """
    asks systemd/timedatectl whether the system clock is currently synchronized
    """
    try:
        result = subprocess.run(
            ['timedatectl', 'show', '-p', 'SystemClockSynchronized', '--value'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().lower() == 'yes'
    except Exception:
        return False

def get_boot_id():
    """
    unique ID for this boot session; changes after reboot
    """
    try:
        with open('/proc/sys/kernel/random/boot_id', 'r') as f:
            return f.read().strip()
    except Exception:
        return 'unknown'
    
def write_status_file(status_path, status_dict):
    """
    writes a single JSON status file atomically so it is never half-written
    """
    payload = dict(status_dict)
    payload['status_written_utc'] = utc_iso_timestamp()

    temp_path = status_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')

    os.replace(temp_path, status_path)

def save_control_values(filename, settings):
    filename += '.txt'
    with open(filename, 'w') as f:
        for k in sorted(settings.keys()):
            f.write('%s: %s\n' % (k, str(settings[k])))
    print('Camera settings saved to %s' % filename) # this whole block creates a .txt file capturing the settings of the camera as it goes

def log(message, log_path):
    """
    prints to terminal AND appends to the flight log file so nothing is lost
    one log file per run, every event goes in it
    """
    line = '[%s] %s' % (utc_iso_timestamp(), message)
    print(line)
    with open(log_path, 'a') as f:
        f.write(line + '\n')

# camera connection check
num_cameras = asi.get_num_cameras()
cameras_found = asi.list_cameras()

if num_cameras == 0:
    print('No camera connected, check USB')
    sys.exit(0)
print('Camera found: %s' % cameras_found[0]) # checking if camera is even connected (if this doesn't pass we are cooked)

camera = asi.Camera(0)
camera_info = camera.get_camera_property()
controls = camera.get_controls()


camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, controls['BandWidth']['MinValue'])
camera.disable_dark_subtract()
camera.set_control_value(asi.ASI_FLIP, 0)

camera.set_roi(width=camera_info['MaxWidth'], height=camera_info['MaxHeight'], bins=1, image_type=asi.ASI_IMG_RAW16) # region of interest
# Make the autoexposure apply to the width and height of 1504 and 1204 respectively, while capturing the entire spectrum so maxheight and width
os.makedirs(save_directory, exist_ok=True)
print('Saving capture to: %s' % save_directory)

# important function definition for adjusting exposure and gain? when in flight
def settle_exposure(camera, log_path, gain = gain, target_mean = target_mean, tolerance = tolerance, 
                        exposure = exposure, exp_min = exp_min, exp_max = exp_max, roi_width = 1504, roi_height = 1204):
    """
    take image with set values and for 10 attempts, try to
    calibrate to get to desired target mean within tolerance and 
    use the exposure for the real image at that target mean
    """
    # safety reset
    try:
        camera.stop_exposure()
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass
    # something might be happening here where it messes the settings INVESTIGATE 
    
    time.sleep(0.5) # buffer time

    camera.set_control_value(asi.ASI_GAIN, gain, auto=False)
    camera.set_image_type(asi.ASI_IMG_RAW16) 

    # this loop is for scaling the exposure to the target mean proportionally
    for attempt in range(15):
        camera.set_control_value(asi.ASI_EXPOSURE, exposure, auto=False)
        frame = camera.capture()
        h, w = frame.shape[:2]
        crop_w = min(roi_width, w)
        crop_h = min(roi_height, h)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        x1 = x0 + crop_w
        y1 = y0 + crop_h
        exposure_region = frame[y0:y1, x0:x1]
        mean = exposure_region.mean()
        log(
            'Exposure attempt %d: exposure=%.1fms mean=%.0f target=%d '
            'region=%dx%d fullframe=%dx%d' %
            (attempt + 1, exposure / 1000, mean, target_mean,
             crop_w, crop_h, w, h),log_path)
        if abs(mean - target_mean) < tolerance:
            break
        scale = target_mean / max(mean, 1)
        scale = max(0.25, min(scale, 4.0))
        exposure = int(exposure * scale)
        exposure = max(exp_min, min(exposure, exp_max))

    return exposure, gain

# takes single image
def capture_frame(camera, exposure, gain):
    # set camera to calibrated settings
    camera.set_control_value(asi.ASI_GAIN, gain, auto=False)
    camera.set_control_value(asi.ASI_EXPOSURE, exposure, auto=False)

    frame = camera.capture()
    return frame

# save raw frame and metadata for information after flight
def save_frame(frame, exposure, gain, save_directory, log_path):
    os.makedirs(save_directory, exist_ok=True)

    timestamp_file = utc_file_timestamp()
    timestamp_iso = utc_iso_timestamp()
    unix_time = time.time()  
    mono_ns = time.monotonic_ns()  
    sync_ok = clock_synchronized()
    boot_id = get_boot_id()  

    base_name = f'{timestamp_file}_exp{exposure}_gain{gain}'

    frame_mean = float(frame.mean())
    frame_max = int(frame.max())

    # save raw data to .fits file (uint16 because yes)
    hdu = fits.PrimaryHDU(frame.astype(np.uint16))

    # metadata - record camera temp
    hdr = hdu.header
    hdr['DATE-OBS'] = (timestamp_iso, 'UTC timestamp ISO-8601')
    hdr['UTCFILE'] = (timestamp_file, 'UTC timestamp YYYYmmdd_HHMMSS')
    hdr['TSUNIX'] = (float(unix_time), 'Best available Unix time [s]')
    hdr['MONONS'] = (str(mono_ns), 'Monotonic ns since boot')
    hdr['CLKSYNC'] = (int(sync_ok), '1 if system clock synchronized')
    hdr['BOOTID'] = (boot_id, 'Linux boot session ID')
    hdr['CAMRTEMP'] = (camera.get_control_value(asi.ASI_TEMPERATURE), 'Camera temperature in Celsius')
    hdr['RSPITEMP'] = (CPUTemperature(), 'Raspberry Pi temperature in Celsius')

    hdr['EXPTIME'] = (exposure, 'Exposure time [microseconds]')
    hdr['EXPTMS'] = (exposure / 1000, 'Exposure time [milliseconds]')
    hdr['GAIN'] = (gain, 'Camera gain setting')
    hdr['FRMROW'] = (frame.shape[0], 'Frame rows')
    hdr['FRMCOL'] = (frame.shape[1], 'Frame columns')
    hdr['FMEAN'] = (frame_mean, 'Mean pixel value')
    hdr['FMAX'] = (frame_max, 'Max pixel value')
    if frame_mean < 5000:
        hdr['FLAG1'] = 'SUSPECT_DARK - possible cloud or something'
    if frame_max >= 60000:
        hdr['FLAG2'] = 'NEAR_SATURATED - lower target_mean or exp_max'

    frame_path = os.path.join(save_directory, base_name + '.fits')
    fits.HDUList([hdu]).writeto(frame_path, overwrite=True)

    # keeping other thing I made before for now 
    meta_path = os.path.join(save_directory, base_name + '.txt')
    with open(meta_path, 'w') as f:
        f.write(f'utc_timestamp_iso: {timestamp_iso}\n')  
        f.write(f'utc_timestamp_file: {timestamp_file}\n')  
        f.write(f'unix_time_s: {unix_time:.6f}\n')
        f.write(f'monotonic_ns: {mono_ns}\n')  
        f.write(f'clock_synchronized: {int(sync_ok)}\n')  
        f.write(f'boot_id: {boot_id}\n')  
        f.write(f'exposure_us: {exposure}\n')
        f.write(f'exposure_ms: {exposure / 1000:.2f}\n')
        f.write(f'gain: {gain}\n')
        f.write(f'frame_shape: {frame.shape}\n')
        f.write(f'frame_mean: {frame_mean:.1f}\n')
        f.write(f'frame_max: {frame_max}\n')
        if frame_mean < 5000:
            f.write('flag: SUSPECT_DARK, possible cloud or something\n')
        if frame_max >= 60000:
            f.write('flag: NEAR_SATURATED, consider lowering target_mean or exp_max\n')

    log('Saved: %s  (mean=%.0f, max=%d)' % (base_name, frame_mean, frame_max), log_path)

    frame_stats = {
        'utc_timestamp_iso': timestamp_iso,
        'utc_timestamp_file': timestamp_file,
        'unix_time_s': unix_time,
        'monotonic_ns': mono_ns,
        'clock_synchronized': bool(sync_ok),
        'boot_id': boot_id,
        'frame_mean': frame_mean,
        'frame_max': frame_max,
        'frame_rows': int(frame.shape[0]),
        'frame_cols': int(frame.shape[1]),
        'frame_path': frame_path,
        'meta_path': meta_path
    }
    return frame_path, meta_path, frame_stats

# in case of cable jiggle or some reason the camera is disconnected during flight
def reconnect_camera(log_path, retries = 10, wait = 5):
    """
    tries to reconnect camera if cable jiggle happens
    """
    print("Attempting to reconnect camera.")
    for attempt in range(retries):
        try:
            # check if camera is visible to the SDK 
            if asi.get_num_cameras() == 0:
                log('Reconnect attempt %d/%d: no camera found, waiting %ds' % (attempt + 1, retries, wait), log_path)
                time.sleep(wait)
                continue

            # rebuild camera settings
            cam = asi.Camera(0)
            info = cam.get_camera_property()
            ctrls = cam.get_controls()
            cam.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, ctrls['BandWidth']['MinValue'])
            cam.disable_dark_subtract()
            cam.set_control_value(asi.ASI_FLIP, 0)
            cam.set_roi(width=info['MaxWidth'], height=info['MaxHeight'], bins=1, image_type=asi.ASI_IMG_RAW16)
            log('Reconnect attempt %d/%d: camera back online' % (attempt + 1, retries), log_path)
            return cam

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            log('Reconnect attempt %d/%d failed: %s, retrying in %ds' % (attempt + 1, retries, str(e), wait), log_path)
            time.sleep(wait)

    log('Camera did not reconnect after %d attempts, retrying next cycle' % retries, log_path)
    return None

def flight_loop(camera, save_directory=save_directory, cadence=cadence):
    """
    this should be almost fully automatic
    """
    os.makedirs(save_directory, exist_ok=True)

    # one log file per run
    log_path = os.path.join(save_directory, 'flight_log_%s.txt' % utc_file_timestamp())
    status_path = os.path.join(save_directory, status_filename)

    log('\nFLIGHT LOOP STARTING', log_path)
    log('Cadence: %ds | Target mean: %d | Gain: %d | Saving to: %s\n'
          % (cadence, target_mean, gain, save_directory), log_path)

    current_exposure = exposure  # updates each cycle
    active_camera = camera
    cycle_count = 0 

    write_status_file(status_path, {
        'state': 'starting',
        'cycle_count': cycle_count,
        'camera_connected': active_camera is not None,
        'current_exposure_us': current_exposure,
        'current_gain': gain,
        'clock_synchronized': clock_synchronized(),
        'boot_id': get_boot_id(),
        'last_error': ''
    })

    while True:
        cycle_count += 1 
        cycle_start = time.time()

        # check for failing camera
        if active_camera is None:
            active_camera = reconnect_camera(log_path)
            if active_camera is None:
                log('Camera failed to reconnect, sleeping then trying again.', log_path)
                write_status_file(status_path, {
                    'state': 'camera_reconnect_failed',
                    'cycle_count': cycle_count,
                    'camera_connected': False,
                    'current_exposure_us': current_exposure,
                    'current_gain': gain,   
                    'clock_synchronized': clock_synchronized(),
                    'boot_id': get_boot_id(),
                    'last_error': 'camera failed to reconnect'
                })
                time.sleep(cadence)
                continue

        try:
            # re-settle every cycle
            current_exposure, current_gain = settle_exposure(active_camera, log_path, gain=gain, exposure=current_exposure)

            frame = capture_frame(active_camera, current_exposure, current_gain)

            fpath, mpath, frame_stats = save_frame(frame, current_exposure, current_gain, save_directory, log_path)

            write_status_file(status_path, {
                'state': 'ok',
                'cycle_count': cycle_count,
                'camera_connected': True,
                'current_exposure_us': current_exposure,
                'current_gain': current_gain,
                'clock_synchronized': frame_stats['clock_synchronized'],
                'boot_id': frame_stats['boot_id'],
                'last_error': '',
                'last_frame_utc': frame_stats['utc_timestamp_iso'],
                'last_frame_path': frame_stats['frame_path'],
                'last_meta_path': frame_stats['meta_path'],
                'last_frame_mean': frame_stats['frame_mean'],
                'last_frame_max': frame_stats['frame_max'],
                'last_frame_shape': [frame_stats['frame_rows'], frame_stats['frame_cols']]
            })

        

        except (KeyboardInterrupt, SystemExit):
            log('Interrupted either by keyboard or system', log_path)
            write_status_file(status_path, {
                'state': 'stopped',
                'cycle_count': cycle_count,
                'camera_connected': active_camera is not None,
                'current_exposure_us': current_exposure,
                'current_gain': gain,
                'clock_synchronized': clock_synchronized(),
                'boot_id': get_boot_id(),
                'last_error': ''
            })
            break
        except Exception as e:
            # log the error but do NOT exit
            log('Error this cycle, skipping and continuing: %s' % str(e), log_path)
            write_status_file(status_path, {
                'state': 'cycle_error',
                'cycle_count': cycle_count,
                'camera_connected': False,
                'current_exposure_us': current_exposure,
                'current_gain': gain,
                'clock_synchronized': clock_synchronized(),
                'boot_id': get_boot_id(),
                'last_error': str(e)
            })
            active_camera = None

        # sleep only the remaining cadence time — settle/capture time counts toward it
        elapsed = time.time() - cycle_start
        sleep_time = max(0, cadence - elapsed)
        log('Cycle: %.1fs elapsed, sleeping %.1fs\n' % (elapsed, sleep_time), log_path)
        time.sleep(sleep_time)

# test for functions
flight_loop(camera)

camera.close()