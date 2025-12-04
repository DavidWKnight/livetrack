from typing import List
from datetime import datetime, timedelta

import pymap3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, constants

from ACState import ACState
from Frame import Frame
from RadReturn import RadReturn
import util

class Scan():
    magData: np.ndarray
    settings: dict # Not an int...
    tStart: float
    targetTimes: List[float]
    targets: List[ACState]

    def __init__(self, data, settings, tStart):
        self.magData = np.absolute(data)
        self.settings = settings
        self.tStart = tStart
        self.targetTimes = []
        self.targets = []

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
        self.targets.append(target)
        return True

    def getTargetLocations(self) -> List[np.ndarray]:
        targets = []
        for i in range(len(self.targets)):
            tProp = self.settings['tStart'] + timedelta(seconds= self.tStart + self.targetTimes[i])
            targetLLA = self.targets[i].getPosition(tProp, True)
            if targetLLA is None:
                continue
            targets.append(targetLLA)
        return targets

    def getDataTimes(self) -> np.ndarray:
        """Return the time that each scan data point is at."""
        sampleRate = self.settings['sampleRate']
        idxs = np.array(range(0, len(self.magData)))
        tValues = self.tStart + (idxs/sampleRate)
        return tValues

    def getMag(self) -> np.ndarray:
        return self.magData

    def getLength(self) -> float:
        return len(self.magData) / self.settings['sampleRate']

    def getFramesTimes(self) -> np.ndarray:
        magData = self.getMag()
        threshold = np.percentile(magData, 99.3)
        mask = magData > threshold
        mask = ndimage.binary_dilation(mask, [True, True, True, True, True])
        clusters, numClusters = ndimage.label(mask)
        centers = np.array([c[0] for c in ndimage.center_of_mass(magData, clusters, range(1,numClusters+1))])
        return (centers / self.settings['sampleRate'], threshold)

    def getFramesTimesCorrected(self, modifyTimes=True) -> np.ndarray:
        centers, threshold = self.getFramesTimes()
        centers = centers * self.settings['sampleRate']

        # Correct pulse times about what we know about the transmitter
        # Note that the pulse times seem so specific that they may vary with time.
        # It's possible we need to infer intervals by looking at the scan data instead of hard coding.
        shortPulse = int(util.ASR9_SHORT_PULSE_INTERVAL * self.settings['sampleRate'])
        longPulse = int(util.ASR9_LONG_PULSE_INTERVAL * self.settings['sampleRate'])
        pulseIdxs = [centers[0]]
        for idx, c in enumerate(centers[1:]):
            interval = c - pulseIdxs[-1]
            if interval < shortPulse*0.9:
                continue # Not a valid pulse
            elif interval > shortPulse*0.9 and interval < shortPulse*1.1:
                if modifyTimes:
                    pulseIdxs.append(pulseIdxs[-1] + shortPulse)
                else:
                    pulseIdxs.append(c)
            elif interval > longPulse*0.95 and interval < longPulse*1.1:
                if modifyTimes:
                    pulseIdxs.append(pulseIdxs[-1] + longPulse)
                else:
                    pulseIdxs.append(c)
            else:
                # Likely didn't pick up the next pulse in the thresholding
                # Try to reset by cenetering the next pulse on the exact next center
                pulseIdxs.append(c)
                # TODO: The -directPulseDelay probably won't be exactly right for any index after this one
                # There should be an increased delay since this likely isn't the direct path
                # Could maybe come up with a map of likely delays vs azimuth angle based on nearby geography
        
        # We want to relate pulses to the transmit time, so need to subtract off the transmit delay
        receiverECEF = np.array(pymap3d.geodetic2ecef(*self.settings['receiverLLA']))
        transmitterECEF = np.array(pymap3d.geodetic2ecef(*self.settings['transmitterLLA']))
        directPulseDelay = np.linalg.norm(receiverECEF - transmitterECEF) / constants.speed_of_light

        pulseTimes = (np.array(pulseIdxs) / self.settings['sampleRate']) - directPulseDelay
        pulseTimes = np.clip(pulseTimes, 0, None)
        # pulseTimes = centers / self.settings['sampleRate'] # For uncorrected
        # pulseTimes = pulseTimes - directPulseDelay
        return (pulseTimes, threshold)

    def toFrames(self, useCorrectedTimes=True, modifyTimes=True) -> List[Frame]:
        # Get the start time of each frame according to pulse locations
        if useCorrectedTimes:
            pulseTimes, _ = self.getFramesTimesCorrected(modifyTimes)
        else:
            pulseTimes, _ = self.getFramesTimes()

        # Convert to indexes and don't forget the first and last frame which are shorter than the others
        frameIndexes = np.int64(np.round(pulseTimes*self.settings['sampleRate']))
        frameIndexes = np.append([0], frameIndexes)
        frameIndexes = np.append(frameIndexes, [len(self.magData)])

        # Chunk up the raw data to create the frames
        tScan = self.getLength()
        frames = []
        for start, end in zip(frameIndexes[:-1], frameIndexes[1:]):
            frameData = self.magData[start:end]
            tFrameStart = start/self.settings['sampleRate']
            az = (tFrameStart / tScan) * 360
            frames.append(Frame(frameData, az, tFrameStart, self.settings))

        return frames

    def applyMatchedFilter(self, kernel=None):
        if kernel is None:
            pulseWidthSamples = int(np.ceil(util.ASR9_PULSE_WIDTH * self.settings['sampleRate']))
            kernel = np.ones(pulseWidthSamples)
        self.magData = np.convolve(self.magData, kernel)

    def applyPulseIntegration(self, nIntegrations=4):
        frames = self.toFrames(True, False)

        outData = np.zeros(len(self.magData))
        idxFrameStart = 0
        for i in range(0, len(frames)):
            # if i % 100 == 0:
            #     print(f"Summing frame {i} out of {len(frames)}")
            
            a = frames[i].magData
            for j in range(1, nIntegrations):
                if i-j >= 0:
                    a = util.padSum(a, frames[i-j].magData)
            
            outData[idxFrameStart:idxFrameStart+len(a)] = a
            idxFrameStart = idxFrameStart + len(a)

            # if True and i % 100 == 0:
            #     for j in range(nIntegrations):
            #         plt.plot(frames[i-j].magData, label=f'{-j}')
            #     plt.plot(a, label='Full')
            #     plt.legend()
            #     plt.show()
        self.magData = outData

    def applyMovingTargetIndicator(self, previousScan):
        frameTimes, _ = self.getFramesTimesCorrected(False)
        prevFrameTimes, _ = previousScan.getFramesTimesCorrected(False)

        for tStart, tEnd in zip(frameTimes[:-1], frameTimes[1:]):
            prevStartTimeIdx = util.nearestIdx(prevFrameTimes, tStart)
            prevStart = prevFrameTimes[prevStartTimeIdx]
            prevEnd = prevFrameTimes[prevStartTimeIdx+1]
            prevStartIdx = int(prevStart * previousScan.settings['sampleRate'])
            prevEndIdx = int(prevEnd * previousScan.settings['sampleRate'])

            startIdx = int(tStart * previousScan.settings['sampleRate'])
            endIdx = int(tEnd * previousScan.settings['sampleRate'])

            prevData = previousScan.magData[prevStartIdx:prevEndIdx]
            mtiData = util.padSub(self.magData[startIdx:endIdx], prevData)
            mtiData = np.clip(mtiData, 0, None)
            if False:
                plt.plot(self.magData[startIdx:endIdx], label='Original')
                plt.plot(prevData, label='Previous')
                plt.plot(mtiData, label='After')
                plt.legend()
                plt.show()

            self.magData[startIdx:endIdx] = mtiData

    def getAllReturns(self, downsample=1) -> list[RadReturn]:
        frames = self.toFrames()
        [_, _, rDirect] = pymap3d.geodetic2aer(*self.settings['receiverLLA'], *self.settings['transmitterLLA'])
        tDirect = rDirect / constants.speed_of_light
        tMaxDist = 100e3 / constants.speed_of_light
        idxDirect = int(tDirect*self.settings['sampleRate'])
        idxMax = int(tMaxDist*self.settings['sampleRate'])

        rets = []
        for i in range(0, len(frames), downsample):
            print(f"Getting returns from time t = {round(frames[i].tStart, 2)}")
            rets.extend(frames[i].getReturns(idxDirect, idxMax))
        return rets

    def plotNearestTargetFrames(self):
        frameTimes, _ = self.getFramesTimesCorrected()
        frames = self.toFrames()

        targetLLAs = []
        for i in range(len(self.targets)):
            tProp = self.settings['tStart'] + timedelta(seconds= self.tStart + self.targetTimes[i])
            targetLLA = self.targets[i].getPosition(tProp)
            targetLLAs.append(targetLLA)

        closestFrameIdx = np.abs(self.tStart + frameTimes - self.targetTimes[i]).argmin()
        targetFrame = frames[closestFrameIdx]
        targetFrame.plotWithTargets(targetLLAs)


