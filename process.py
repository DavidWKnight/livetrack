from datetime import timedelta
import time
import cProfile

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
# fnameBase = '2025-10-06T17:26:46_AC98AC'
# fnameBase = '2025-10-06T17:06:14_A609D4'
fnameBase = '2025-10-06T16:54:34_A05F76'
# fnameBase = '2025-10-06T16:57:22_A05F76'


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

previousScan = None
while not RFDataManager.isEndOfFile():
    allReturns = []
    for j in range(2):
        scan = RFDataManager.getNextScan()
        
        scan.applyMatchedFilter()

        scan.applyPulseIntegration(10)
        scan.appendTarget(aircraftState)
        if previousScan is not None:
            # scan.applyMovingTargetIndicator(previousScan)
            pass
        # scan.plotNearestTargetFrames()
        frames = scan.toFrames()

        # amplitudeSpectrogramPlot(scan, settings)

        tStart = time.time()

        # cProfile.run('returns = scan.getAllReturns(100)', sort='tottime')
        # input('')
        returns = scan.getAllReturns(100)

        # Check for additional frames near the target
        targetLLAs = scan.getTargetLocations()
        for targetLLA in targetLLAs:
            [az, el, srange] = pymap3d.geodetic2aer(*targetLLA, *settings['transmitterLLA'])
            print(f"targetLLA = {targetLLA}, {[az, el, srange]}")
            for frame in frames:
                if abs(frame.az - az) > 0.5:
                    continue
                # frame.plotWithTargets([targetLLA])
                newReturns = frame.getReturns()
                for r in newReturns:
                    [_, _, distError] = pymap3d.geodetic2aer(*r.LLA, *targetLLA)
                    if distError > 5e3:
                        continue
                    print(f"distError = {distError}")
                returns.extend(newReturns)

        allReturns.append(returns)

        previousScan = scan
    

    plotReturns(allReturns, targetLLAs, scan.settings)
