from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

import dataLoad

from plotData import amplitudeSpectrogramPlot
from plotData import amplitudePlot
from plotData import plotReturns

folder = '/media/david/Ext/Collects/**/'
# fnameBase = '2025-10-04T12:08:50_ABC504'
# fnameBase = '2025-09-28T17:06:03_A3AA32'
fnameBase = '2025-10-06T17:26:46_AC98AC'
# fnameBase = '2025-10-06T17:22:43_ABBD0A'
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
        scan.appendTarget(aircraftState)
        scan.plotNearestTargetFrames()
        frames = scan.toFrames()

        amplitudeSpectrogramPlot(scan, settings)

        returns = []
        for i in range(0, len(frames), 100):
            print(f"Getting returns from time t = {round(frames[i].tStart, 2)}")
            returns.extend(frames[i].getReturns())
        allReturns.append(returns)
    
    targets = []
    for i in range(len(scan.targets)):
        tProp = settings['tStart'] + timedelta(seconds= scan.tStart + scan.targetTimes[i])
        targetLLA = scan.targets[i].getPosition(tProp)
        targets.append(targetLLA)

    plotReturns(allReturns, targets, scan.settings)
