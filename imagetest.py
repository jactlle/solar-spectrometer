import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits 

# frame = np.load('/home/jacobt/spectrometer/captures/20260609_140235_exp18496_gain0.fits')

with fits.open('/home/jacobt/spectrometer/captures/20260609_140235_exp18496_gain0.fits') as hdul:
    frame = hdul[0].data        # numpy uint16 array
    header = hdul[0].header     # metadata baked into the file


print('Shape:', frame.shape)
print('Mean:', frame.mean())
print('Max:', frame.max())
print('Exposure (us):', header['EXPTIME'])
print('Gain:', header['GAIN'])
print('Timestamp:', header['TIMESTMP'])


plt.imshow(frame, cmap='gray', aspect='auto', vmin=0, vmax=65535)
plt.colorbar()
plt.title('Raw spectrum frame')
plt.savefig('test_frame.png', dpi=100, bbox_inches='tight')
plt.close()
print('Saved test_frame.png')