from typing import List
from datetime import datetime, timedelta

import numpy as np
import pymap3d

from ACState import ACState
from radarAlgo import findDirectPathPulses

class Return():
    AER: np.ndarray
    t: datetime

    def __init__(self):
        pass


class Frame():
    def __init__(self):
        pass

    def getReturns(self):
        # Perform CFAR, take CFAR parameters as args
        pass

class Scan():
    data: np.ndarray
    settings: int
    tStart: float
    targetTimes = List[float]

    def __init__(self, data, settings, tStart):
        self.data = data
        self.settings = settings
        self.tStart = tStart
        self.targetTimes = []

    def appendTarget(self, target: ACState) -> bool:
        transmitterLLA = self.settings['transmitterLLA']
        receiverLLA = self.settings['receiverLLA']
        recieverAER = pymap3d.geodetic2aer(*receiverLLA, *transmitterLLA)
        
        tDet = 0 # Initial guess
        nIterations = 3

        collectStart = self.settings['tStart']

        for _ in range(nIterations):
            tProp = collectStart + timedelta(seconds= self.tStart + tDet)
            targetLLA = target.getPosition(tProp)
            if targetLLA is None:
                return False
            aircraftAER = pymap3d.geodetic2aer(*targetLLA, *transmitterLLA)
            dAz = aircraftAER[0] - recieverAER[0]
            if dAz < 0:
                dAz = dAz + 360
            # Could early return if dAz is small
            tDet = (dAz/360) * 4.6
        
        self.targetTimes.append(self.tStart + tDet)
        return True

    def getDataTimes(self) -> np.ndarray:
        """Return the time that each scan data point is at."""
        sampleRate = self.settings['sampleRate']
        idxs = np.array(range(0, len(self.data)))
        tValues = self.tStart + (idxs/sampleRate)
        return tValues

    def getMag(self) -> np.ndarray:
        return np.absolute(self.data)


def collect2scans(data: np.ndarray, settings) -> List[Scan]:
    sampleRate = settings['sampleRate']
    try:
        directPath, directPathIdx = findDirectPathPulses(data, sampleRate, 1, False)
    except:
        # Can't find pulses, just plot with no offset
        print(f"Unable to find direct pulses, aligning to start of collect")
        tCollect = len(data) / sampleRate
        nFullScans = int(np.floor(tCollect / 4.6))
        directPath = np.array(range(nFullScans))*4.6
        directPathIdx = directPath * sampleRate

    scans = []
    for idxStart, idxEnd in zip(directPathIdx[:-1], directPathIdx[1:]):
        scanData = data[idxStart:idxEnd]
        tStart = idxStart / sampleRate
        scans.append(Scan(scanData, settings, tStart))

    return scans
