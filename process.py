import copy
import time
import glob
import csv
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
import pymap3d

import dataLoad
from radarAlgo import maxPool1d, findDirectPathPulses, pulseIntegration, matchFilter
from ACState import ACPosition

def lin2db(x):
    return 20 * np.log10(x)

folder = '/home/david/Documents/Grad_School/Bistatic-Radar/Collects/collect_9_28/collects4/'
fnameBase = '2025-09-28T17:10:12_A0851B'
# fnameBase = '2025-09-28T17:06:03_A3AA32'
# fnameBase = '2025-09-28T17:02:31_A3FBA4'
transmitterLLA = np.array([34.052724, -117.596634, 0])
receiverLLA = [34.051555, -117.593415, 0]
recieverAER = pymap3d.geodetic2aer(*receiverLLA, *transmitterLLA)

aircraftState, data = dataLoad.loadCollect(folder + fnameBase)
data = data[4:] # Remove header
data = data[::2] + -1j * data[1::2]
data = np.absolute(data)

# data = np.absolute(data[:int(len(data) / 4)])
# data = data - np.mean(data)
# fftData = np.fft.fft(data)
# magResponse = np.absolute(fftData)
# magResponse = lin2db(magResponse)
# plt.plot(magResponse[:int(len(magResponse) / 2)])
# plt.show()

tStart = aircraftState.positions[0].t

sampleRate = 1e6 * 0.75

start = time.time()
# data = matchFilter(data, sampleRate)
data = pulseIntegration(data, sampleRate)
end = time.time()
print(f"pulseIntegration took {end - start} seconds")

directPath, directPathIdx = findDirectPathPulses(data, sampleRate, 1, False)

aircraftDetTime = []
aircraftDetTimeGuess = []
for frameStart in directPath:
    tDet = 0
    nIterations = 2
    for i in range(nIterations):
        tProp = tStart + timedelta(seconds=(frameStart+tDet))
        aircraftLLAFrameStart = aircraftState.getPosition(tProp)
        
        extrap = False
        if aircraftLLAFrameStart is None: # Requested time is outside recorded positions
            extrap = True
            aircraftLLAFrameStart = aircraftState.getPosition(tProp, True)
        
        aircraftAERFrameStart = pymap3d.geodetic2aer(*aircraftLLAFrameStart, *transmitterLLA)
        dAz = aircraftAERFrameStart[0] - recieverAER[0]
        if dAz < 0:
            dAz = dAz + 360
        tDet = (dAz/360) * 4.6

    if extrap:
        aircraftDetTimeGuess.append(frameStart + tDet)
    else:
        aircraftDetTime.append(frameStart + tDet)

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
print(aircraftDetTime)

plt.plot(np.array(range(0, len(data))) * (1/sampleRate), data)
plt.axhline(y=asrThreshold, color='black', linestyle='--')
plt.scatter(directPath, retAmplitude, color='gray')

# Put lines where we expect aircaft
for t in aircraftDetTime:
    plt.axvline(t, color='r', linestyle='--')
for t in aircraftDetTimeGuess:
    plt.axvline(t, color='peru', linestyle='--')

plt.show()
