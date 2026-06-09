#!/usr/bin/env python

import os
import sys
import time # time.sleep() for settling loop
import zwoasi as asi # camera wrapper
import numpy as np
from astropy.io import fits

# variable definitions
cadence = 15 # in seconds
target_mean = 40000 # pixel target for spectra? 
tolerance = 3000 # within 3000 of target? ok
exposure = 100000 # starting guess
exp_min = 1000 # minimum allowed (1 ms)
exp_max = 10000000 # maximum allowed (10 s) may be lowered
gain = 0 # for low light spectra, can change

# TO DO : Figure out what happens to the time when you disconnect wifi, power cycle, and reconnect wifi and figure out a solution

# gathering file names
env_filename = os.path.expanduser('~/zwo/libASICamera2.so') # Need to figure out how to download ZWO_ASI_LIB 
save_directory = os.path.expanduser('~/spectrometer/captures')

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
    line = '[%s] %s' % (time.strftime('%Y%m%d_%H%M%S'), message)
    print(line)
    with open(log_path, 'a') as f:
        f.write(line + '\n')

if not env_filename:
    print('ZWO_ASI_LIB not set, exiting')
    sys.exit(1)
asi.init(env_filename) # this little block is checking if the folder is there and alive (if this doesn't pass we are cooked)

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

# camera setup
camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, controls['BandWidth']['MinValue'])
camera.disable_dark_subtract()
camera.set_control_value(asi.ASI_FLIP, 0)

camera.set_roi(width=camera_info['MaxWidth'], height=200, bins=1, image_type=asi.ASI_IMG_RAW16) # region of interest

os.makedirs(save_directory, exist_ok=True)
print('Saving capture to: %s' % save_directory)

# important function definition for adjusting exposure and gain? when in flight
def settle_exposure(camera, log_path, gain = gain, target_mean = target_mean, tolerance = tolerance, 
                        exposure = exposure, exp_min = exp_min, exp_max = exp_max):
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
    for attempt in range(20):
        camera.set_control_value(asi.ASI_EXPOSURE, exposure, auto=False)
        frame = camera.capture()
        mean = frame.mean()
        log('Exposure attempt %d: exposure=%.1fms  mean=%.0f  target=%d' %
              (attempt + 1, exposure / 1000, mean, target_mean), log_path)
        if abs(mean - target_mean) < tolerance:
            break  # settled or close enough
        # proportional scale: if mean is half the target, double exposure
        # maybe add a scale limiter so it doesnt jump to exposure of 7k from one little dark spot
        scale = target_mean / max(mean, 1)
        scale = max(0.25, min(scale, 4.0)) # no more than 4x jump per attempt
        exposure = int(exposure*scale)
        exposure = max(exp_min, min(exposure, exp_max)) # clamp to valid range
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

    # timestamped for time during flight (idk if important)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_name = f'{timestamp}_exp{exposure}_gain{gain}'

    # save raw data to .fits file (uint16 because yes)
    hdu = fits.PrimaryHDU(frame.astype(np.uint16))

    # metadata
    hdr = hdu.header
    hdr['TIMESTMP'] = (timestamp, 'UTC timestamp YYYYmmdd_HHMMSS')
    hdr['EXPTIME'] = (exposure, 'Exposure time [microseconds]')
    hdr['EXPTMS'] = (exposure / 1000, 'Exposure time [milliseconds]')
    hdr['GAIN'] = (gain, 'Camera gain setting')
    hdr['FRMROW'] = (frame.shape[0], 'Frame rows')
    hdr['FRMCOL'] = (frame.shape[1], 'Frame columns')
    hdr['FMEAN'] = (float(frame.mean()), 'Mean pixel value')
    hdr['FMAX'] = (int(frame.max()), 'Max pixel value')
    if frame.mean() < 5000:
        hdr['FLAG1'] = 'SUSPECT_DARK - possible cloud or something'
    if frame.max() >= 65000:
        hdr['FLAG2'] = 'NEAR_SATURATED - lower target_mean or exp_max'

    frame_path = os.path.join(save_directory, base_name + '.fits')
    fits.HDUList([hdu]).writeto(frame_path, overwrite=True)

    # keeping other thing I made before for now 
    meta_path = os.path.join(save_directory, base_name + '.txt')
    with open(meta_path, 'w') as f:
        f.write(f'timestamp: {timestamp}\n')
        f.write(f'exposure_us: {exposure}\n')
        f.write(f'exposure_ms: {exposure / 1000:.2f}\n')
        f.write(f'gain: {gain}\n')
        f.write(f'frame_shape: {frame.shape}\n')
        f.write(f'frame_mean: {frame.mean():.1f}\n')
        f.write(f'frame_max: {frame.max()}\n')
        if frame.mean() < 5000:
            f.write('flag: SUSPECT_DARK, possible cloud or something\n')
        if frame.max() >= 65000:
            f.write('flag: NEAR_SATURATED, consider lowering target_mean or exp_max\n')

    log('Saved: %s  (mean=%.0f, max=%d)' % (base_name, frame.mean(), frame.max()), log_path)
    return frame_path, meta_path

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
            cam.set_roi(width=info['MaxWidth'], height=200, bins=1, image_type=asi.ASI_IMG_RAW16)
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
    log_path = os.path.join(save_directory, 'flight_log_%s.txt' % time.strftime('%Y%m%d_%H%M%S'))

    log('\nFLIGHT LOOP STARTING', log_path)
    log('Cadence: %ds | Target mean: %d | Gain: %d | Saving to: %s\n'
          % (cadence, target_mean, gain, save_directory), log_path)

    current_exposure = exposure  # updates each cycle
    active_camera = camera

    while True:
        cycle_start = time.time()

        # check for failing camera
        if active_camera is None:
            active_camera = reconnect_camera(log_path)
            if active_camera is None:
                log('Camera failed to reconnect, sleeping then trying again.', log_path)
                time.sleep(cadence)
                continue

        try:
            # re-settle every cycle
            current_exposure, current_gain = settle_exposure(camera, log_path, gain=gain, exposure=current_exposure)

            frame = capture_frame(camera, current_exposure, current_gain)

            fpath, mpath = save_frame(frame, current_exposure, current_gain, save_directory, log_path)

        except (KeyboardInterrupt, SystemExit):
            log('Interrupted either by keyboard or system', log_path)
            break
        except Exception as e:
            # log the error but do NOT exit
            log('Error this cycle, skipping and continuing: %s' % str(e), log_path)

        # sleep only the remaining cadence time — settle/capture time counts toward it
        elapsed = time.time() - cycle_start
        sleep_time = max(0, cadence - elapsed)
        log('Cycle: %.1fs elapsed, sleeping %.1fs\n' % (elapsed, sleep_time), log_path)
        time.sleep(sleep_time)

# test for functions
flight_loop(camera)

camera.close()