from typing import List
import copy
import json

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

    def getReturns(self, startIdx=0, endIdx=None, targetEl=None, plot=None) -> List[RadReturn]:
        # Perform CFAR, take CFAR parameters as args
        procData = self.magData[startIdx:endIdx]
        [cfar_values, targets_only] = cfar(procData, 3, 3, 3)
        mask = ndimage.binary_erosion(targets_only)
        clusters, numClusters = ndimage.label(mask)
        if numClusters == 0:
            return []
        centers = np.array([c[0] for c in ndimage.center_of_mass(procData, clusters, range(1,numClusters+1))])

        if plot is not None:
            targets_only[~mask] = 0
            plt.plot(targets_only, label='MagData Eroded')
            plt.plot(cfar_values, label='cfar')
            plt.legend()
            plt.show()
        centers = centers + startIdx # Add back in the part we skipped
        branges = (centers / self.settings['sampleRate']) * constants.speed_of_light

        minRange = 500 # Meters
        maxRange = 40e3 # Meters
        branges = branges[branges > minRange]
        returns = []
        initialElEst = 3

        with open("elProfile.json", 'r') as inFile:
            profileIdx = int(np.floor(self.az))
            elProfile = np.poly1d(json.load(inFile)[profileIdx])

        for br in branges:
            # Start with an assumption elevation angle
            r = bistaticRange2ElRange(self.settings['transmitterLLA'], self.settings['receiverLLA'], br, self.az, initialElEst)
            # Then infer an elevation angle from the profile
            testENU = pymap3d.aer2enu(self.az, initialElEst, r)
            el = elProfile(np.linalg.norm(testENU[0:2]))
            r = bistaticRange2ElRange(self.settings['transmitterLLA'], self.settings['receiverLLA'], br, self.az, el)

            if r is None:
                continue

            targetLLA = pymap3d.aer2geodetic(self.az, el, r, *self.settings['transmitterLLA'])
            returns.append(RadReturn(targetLLA, self.tStart))
        return returns

    def plotWithTargets(self, targetLLAs):
        [_, _, receiveRange] = pymap3d.geodetic2aer(*self.settings['receiverLLA'], *self.settings['transmitterLLA'])
        receiveIdx = (receiveRange / constants.speed_of_light)*self.settings['sampleRate']

        targetIndexes = []
        for targetLLA in targetLLAs:
            [_, _, srangeTransmitter] = pymap3d.geodetic2aer(*targetLLA, *self.settings['transmitterLLA'])
            [_, _, srangeReciever] = pymap3d.geodetic2aer(*targetLLA, *self.settings['receiverLLA'])
            srange = srangeTransmitter + srangeReciever
            rangeTime = srange / constants.speed_of_light
            rangeIdx = int(rangeTime * self.settings['sampleRate'])
            targetIndexes.append(rangeIdx)
        
        plt.plot(self.magData, label='magData')
        plt.axvline(receiveIdx, color='pink', label='receiver')
        for targetIdx in targetIndexes:
            plt.axvline(targetIdx, color='red', label='target')

        [cfar_values, _] = cfar(self.magData, 3, 3, 3)
        plt.plot(cfar_values, label='cfar')
        plt.legend()

        plt.title(f'Nearst target frame at az {round(self.az,1)}')
        plt.show()

    def getMag(self):
        return self.magData

    def __add__(self, other):
        result = copy.copy(self.magData)
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
        result = copy.copy(self.magData)
        otherIsShorter = len(other.magData) < len(self.magData)
        otherIsLonger = len(other.magData) > len(self.magData)

        if otherIsShorter:
            result[:len(other.magData)] = result[:len(other.magData)] - other.magData
        elif otherIsLonger:
            result = result - other.magData[:len(self.magData)]
        else:
            result = result - other.magData
        
        return Frame(result, self.az, self.tStart, self.settings)
