from typing import List

import pymap3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, constants

from radarAlgo import cfar, bistaticRange2ElRange

from RadReturn import RadReturn

class Frame():
    def __init__(self, data, az, tStart, settings):
        self.magData = np.absolute(data)
        self.az = az
        self.tStart = tStart
        self.settings = settings

    def getReturns(self) -> List[RadReturn]:
        # Perform CFAR, take CFAR parameters as args
        [cfar_values, targets_only] = cfar(self.magData, 3, 3, 3)
        mask = ndimage.binary_erosion(targets_only)
        clusters, numClusters = ndimage.label(mask)
        centers = np.array([c[0] for c in ndimage.center_of_mass(self.magData, clusters, range(1,numClusters+1))])

        if False:
            targets_only[~mask] = 0
            plt.plot(targets_only, label='MagData Eroded')
            plt.plot(cfar_values, label='cfar')
            plt.legend()
            plt.show()

        branges = (centers / self.settings['sampleRate']) * constants.speed_of_light

        minRange = 500 # Meters
        maxRange = 40e3 # Meters
        branges = branges[branges > minRange]
        returns = []
        for r in branges:
            r, el = bistaticRange2ElRange(self.settings['transmitterLLA'], self.settings['receiverLLA'], r, self.az)
            if r < minRange or r > maxRange:
                continue
            targetLLA = pymap3d.aer2geodetic(self.az, el, r, *self.settings['transmitterLLA'])
            returns.append(RadReturn(targetLLA, self.tStart))
        return returns


    def getMag(self):
        return self.magData

    def __add__(self, other):
        result = self.magData
        otherIsShorter = len(other.magData) < len(self.magData)
        otherIsLonger = len(other.magData) > len(self.magData)

        if otherIsShorter:
            result[:len(other.magData)] = result[:len(other.magData)] + other.magData
        elif otherIsLonger:
            result = result + other.magData[:len(self.magData)]
        else:
            result = result + other.magData
        
        return Frame(result, self.az, self.tStart, self.settings)

    def __sub__(self, other):
        result = self.magData
        otherIsShorter = len(other.magData) < len(self.magData)
        otherIsLonger = len(other.magData) > len(self.magData)

        if otherIsShorter:
            result[:len(other.magData)] = result[:len(other.magData)] - other.magData
        elif otherIsLonger:
            result = result - other.magData[:len(self.magData)]
        else:
            result = result - other.magData
        
        return Frame(result, self.az, self.tStart, self.settings)
