#!/usr/bin/env python

import os
import sys
import time # time.sleep() for settling loop
import zwoasi as asi # camera wrapper
import numpy as np

# for me : use auto_exposure() later from the binning demo perjaps

def save_control_values(filename, settings):
    filename += '.txt'
    with open(filename, 'w') as f:
        for k in sorted(settings.keys()):
            f.write('%s: %s\n' % (k, str(settings[k])))
    print('Camera settings saved to %s' % filename) # this whole block creates a .txt file capturing the settings of the camera as it goes

# gathering file name 
env_filename = os.getenv('ZWO_ASI_LIB')
save_directory = os.path.expanduser('~/spectrometer/captures')
cadence = 30 # in seconds, can definitely change better -> TO DISCUSS

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

camera.set_roi(width=5496, height=200, bins=1, image_type=asi.ASI_IMG_RAW16) # region of interest

os.makedirs(save_directory, exist_ok=True)
print('Saving capture to: %s' % save_directory)

# important function definition for adjusting exposure and gain? when in flight
def settle_exposure(camera, gain = 0, target_mean = 40000, tolerance = 3000, 
                        exposure = 100000, exp_min = 1000, exp_max = 10000000):
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
        print('Exposure attempt %d: exposure=%.1fms  mean=%.0f  target=%d' %
              (attempt + 1, exposure / 1000, mean, target_mean))
        if abs(mean - target_mean) < tolerance:
            break  # settled or close enough
        # proportional scale: if mean is half the target, double exposure
        # maybe add a scale limiter so it doesnt jump to exposure of 7k from one little dark spot
        exposure = int(exposure*(target_mean/max(mean, 1)))
        exposure = max(exp_min, min(exposure, exp_max)) # clamp to valid range
        if exposure < exp_min or mean < 100:
            print('WARNING: settling had garbage values brobro')
    return exposure, gain

# takes single image
def capture_frame(camera, exposure, gain):
    # set camera to calibrated settings
    camera.set_control_value(asi.ASI_GAIN, gain, auto=False)
    camera.set_control_value(asi.ASI_EXPOSURE, exposure, auto=False)

    frame = camera.capture()
    return frame

# save raw frame and metadata for information after flight
def save_frame(frame, exposure, gain, save_directory):
    os.makedirs(save_directory, exist_ok=True)

    # timestamped for time during flight (idk if important)
    timestamp = time.strftime('%H%M%S')
    base_name = f'{timestamp}_exp{exposure}_gain{gain}'

    # save raw data to npy for now 
    frame_path = os.path.join(save_directory, base_name + '.npy')
    np.save(frame_path, frame)

    # metadata
    meta_path = os.path.join(save_directory, base_name + '.txt')
    with open(meta_path, 'w') as f:
        f.write(f'exposure_us: {exposure}\n')
        f.write(f'gain: {gain}\n')
        f.write(f'timestamp: {timestamp}\n')

    return frame_path, meta_path

# test for functions
exp, gain = settle_exposure(camera, gain = 0, target_mean = 40000, tolerance = 3000, 
                        exposure = 100000, exp_min = 1000, exp_max = 10000000)
print('Gain: %d, Exposure: %.1fms' % (gain, exp/1000))
camera.close()