from datetime import timedelta
import time

import numpy as np
import matplotlib.pyplot as plt
import pymap3d
from scipy import ndimage
from scipy import constants

import dataLoad

from plotData import amplitudeSpectrogramPlot
from plotData import amplitudePlot
from plotData import plotReturns

folder = '/media/david/Ext/Collects/**/'
# fnameBase = '2025-10-04T12:08:50_ABC504'
# fnameBase = '2025-09-28T17:06:03_A3AA32'
# fnameBase = '2025-10-06T17:26:46_AC98AC'
fnameBase = '2025-10-06T17:06:14_A609D4'
# fnameBase = '2025-09-28T17:17:35_A420AF'

aircraftState, settings, RFDataManager = dataLoad.loadCollect(folder + fnameBase)

# previous = RFDataManager.getNextScan()
# while not RFDataManager.isEndOfFile():
#     scan = RFDataManager.getNextScan()
#     scan.appendTarget(aircraftState)
    
#     pulseLines, threshold = scan.getFramesTimes()
#     plt.plot(np.diff(pulseLines))
#     plt.show()

#     plt.hist(np.diff(pulseLines), np.arange(0.6e-3, 1.1e-3, 1e-6))
#     plt.show()

#     # scan.getClutterSuppressedMag()

#     amplitudePlot(scan, settings, False)
#     # amplitudePlot(scan.applyPulseIntegration(), settings, False)
#     # amplitudePlot(previous, settings, False)
#     plt.show()

#     previous = scan

while not RFDataManager.isEndOfFile():
    allReturns = []
    for j in range(2):
        scan = RFDataManager.getNextScan()

        scan.applyPulseIntegration()
        scan.appendTarget(aircraftState)
        # scan.plotNearestTargetFrames()
        frames = scan.toFrames()

        # amplitudeSpectrogramPlot(scan, settings)

        tStart = time.time()
        returns = []

        [_, _, rDirect] = pymap3d.geodetic2aer(*settings['receiverLLA'], *settings['transmitterLLA'])
        tDirect = rDirect / constants.speed_of_light
        tMaxDist = 100e3 / constants.speed_of_light
        for i in range(0, len(frames), 100):
            print(f"Getting returns from time t = {round(frames[i].tStart, 2)}")
            returns.extend(frames[i].getReturns(int(tDirect*settings['sampleRate']), int(tMaxDist*settings['sampleRate'])))
        tEnd = time.time()
        print(f"Took {tEnd - tStart} seconds")

        # Check for additional frames near the target
        targetTime = settings['tStart'] + timedelta(seconds=scan.tStart)
        targetLLA = aircraftState.getPosition(targetTime)
        [az, _, _] = pymap3d.geodetic2aer(*targetLLA, *settings['transmitterLLA'])
        for frame in frames:
            if abs(frame.az - az) > 0.5:
                continue
            returns.extend(frame.getReturns())

        allReturns.append(returns)
    
    targets = []
    for i in range(len(scan.targets)):
        tProp = settings['tStart'] + timedelta(seconds= scan.tStart + scan.targetTimes[i])
        targetLLA = scan.targets[i].getPosition(tProp)
        targets.append(targetLLA)

    plotReturns(allReturns, targets, scan.settings)
