import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

frame = np.load('/home/jacobt/spectrometer/captures/20260609_073904_exp3443_gain0.npy')

print('Shape:', frame.shape)
print('Mean:', frame.mean())
print('Max:', frame.max())

plt.imshow(frame, cmap='gray', aspect='auto', vmin=0, vmax=65535)
plt.colorbar()
plt.title('Raw spectrum frame')
plt.savefig('test_frame.png', dpi=100, bbox_inches='tight')  # save to file
plt.close()
print('Saved test_frame.png')