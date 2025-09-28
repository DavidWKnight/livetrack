import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal

from radarAlgo import maxPool1d, findDirectPathPulses
from ACState import ACPosition

def lin2db(x):
    return 10 * np.log10(x)

data = np.fromfile('2025-09-26T13:11:21_A2869F.dat')
# data = np.fromfile('2025-09-26T17:18:47_A3ECC8.dat')
data = data[::2] + -1j * data[1::2]
data = np.absolute(data)

sampleRate = 1e6

directPath, directPathIdx = findDirectPathPulses(data, sampleRate, 1, False)

decimate = 10
data = maxPool1d(data, decimate)
sampleRate = sampleRate / decimate

# data = data[int((sampleRate/decimate)*3):int((sampleRate/decimate)*5)]
data = lin2db(data)

asrThreshold = np.percentile(data, 95)

print(directPath)
print(np.diff(directPath))
retAmplitude = data[np.array(directPathIdx) // decimate]
print(retAmplitude)

plt.plot(np.array(range(0, len(data))) * (1/sampleRate), data)
plt.axhline(y=asrThreshold, color='red', linestyle='--')
plt.scatter(directPath, retAmplitude)

plt.show()
