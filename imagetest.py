import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open('/home/jacobt/spectrometer/captures/20260611_185655_exp25537_gain380.fits') as hdul:
    frame = hdul[0].data
    header = hdul[0].header

print('Shape:', frame.shape)
print('Mean:', frame.mean())
print('Max:', frame.max())
print('Exposure (ms):', header['EXPTIME'] / 1000)
print('Gain:', header['GAIN'])

vmin = np.percentile(frame, 1)
vmax = np.percentile(frame, 99)

plt.figure(figsize=(10, 6))
plt.imshow(frame, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
plt.colorbar(label='ADU')
plt.title(f"Raw spectrum | exp={header['EXPTIME']/1000:.1f}ms | gain={header['GAIN']}")
plt.tight_layout()
plt.savefig('test_frame.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved test_frame.png')