from datetime import timedelta
import time
import cProfile

import numpy as np
import matplotlib.pyplot as plt
import pymap3d
from scipy import ndimage
from scipy import constants

import dataLoad
from PlaneKF import PlaneKF

from plotData import amplitudeSpectrogramPlot
from plotData import amplitudePlot
from plotData import plotReturns

folder = '/media/david/Ext/Collects/**/'

# fnameBase = '2025-10-04T11:17:39_A5CE0F'

# fnameBase = '2025-10-06T16:26:15_AC41F3'
# fnameBase = '2025-10-06T17:26:46_AC98AC'
# fnameBase = '2025-10-06T17:06:14_A609D4'
fnameBase = '2025-10-06T16:54:34_A05F76' # Good to up to ~2.8km when cheating on el angle - 737-7H4
# fnameBase = '2025-10-06T16:57:22_A05F76'
# fnameBase = '2025-10-06T17:20:42_899122'

# fnameBase = '2025-11-23T12:56:19_A2C27E'
# fnameBase = '2025-11-23T13:17:44_AD2FD0'
# fnameBase = '2025-11-23T13:47:32_A46A72'
# fnameBase = '2025-11-23T16:26:55_89900F'


# fnameBase = '2025-11-24T16:35:35_AC5697'
# fnameBase = '2025-11-24T17:04:32_A11EB7'
# fnameBase = '2025-11-24T17:14:59_A1C868'
# fnameBase = '2025-11-24T16:20:02_A562E8'
# fnameBase = '2025-11-24T16:22:17_A90B27'

aircraftState, allAC, settings, RFDataManager = dataLoad.loadCollect(folder + fnameBase)

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
tOffset = 5
tPlaneStart = settings['tStart'] + timedelta(seconds=tOffset)
pInit = aircraftState.getPositionENU(settings['transmitterLLA'], tPlaneStart)
vInit = aircraftState.getPositionENU(settings['transmitterLLA'], tPlaneStart+timedelta(seconds=1)) - aircraftState.getPositionENU(settings['transmitterLLA'], tPlaneStart)
# vInit = aircraftState.getVelocityENU(tPlaneStart)
P = np.diag([5e3, 50, 5e3, 50, 5e3, 25])
pkf = PlaneKF(settings['transmitterLLA'], pInit, vInit, tPlaneStart, P)

previousScan = None
i = 0
while not RFDataManager.isEndOfFile():
    while i < 2:
        scan = RFDataManager.getNextScan()
        i = i + 1
        continue

    allReturns = []
    for j in range(2):
        scan = RFDataManager.getNextScan()
        
        # scan.applyMatchedFilter()

        scan.applyPulseIntegration(10)
        
        scan.appendTarget(aircraftState)
        # for ac in allAC.values():
        #     scan.appendTarget(ac)
        
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
        planeReturns = []
        targetLLAs = scan.getTargetLocations()
        for targetLLA in targetLLAs[:1]:
            [az, el, srange] = pymap3d.geodetic2aer(*targetLLA, *settings['transmitterLLA'])
            print(f"targetLLA = {targetLLA}, {[az, el, srange]}")
            for frame in frames:
                if abs(frame.az - az) > 0.5:
                    continue
                # frame.plotWithTargets([targetLLA])

                estTargetENU = pkf.predictPos(settings['tStart'] + timedelta(seconds=scan.tStart + frame.tStart))
                estTargetAER = pymap3d.enu2aer(*estTargetENU)
                estTargetLLA = pymap3d.enu2geodetic(*estTargetENU, *settings['transmitterLLA'])
                # newReturns = frame.getReturns(targetEl=estTargetAER[1])
                # newReturns = frame.getReturns(targetEl=el)
                newReturns = frame.getReturns()
                if len(newReturns) == 0:
                    continue

                returns.extend(newReturns)

                def getError(r):
                    # [_, _, distError] = pymap3d.geodetic2aer(*r.LLA, *estTargetLLA)
                    [_, _, distError] = pymap3d.geodetic2aer(*r.LLA, *targetLLA)
                    return distError
                distance = list(map(getError, newReturns))

                bestRet = np.argmin(distance)
                if distance[bestRet] < 1500:

                    est = newReturns[bestRet]
                    print(f"best est = {est.AER}")
                    est.t = settings['tStart'] + timedelta(seconds=scan.tStart + est.t) # type: ignore
                    pkf.estimate(est)

        allReturns.append(returns)

        previousScan = scan
    
    plt.figure()
    if pkf is not None:
        error = []
        pkfSaver = pkf.getSaver()
        for idx, t in enumerate(pkf.t[1:]):
            truthPos = aircraftState.getPositionENU(settings['transmitterLLA'], t)
            estPos = pkfSaver.x[idx][::2]
            error.append(np.linalg.norm(truthPos - estPos))
        plt.plot(error)
    
    plt.figure()
    plotReturns(allReturns, targetLLAs, scan.settings)
