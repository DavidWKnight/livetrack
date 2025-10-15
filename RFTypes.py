from typing import List
from datetime import datetime, timedelta

import pymap3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, constants

from ACState import ACState
from radarAlgo import findDirectPathPulses, cfar, bistaticRange2ElRange

ASR11_ROTATION_RATE = 60/13

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class Return():
    AER: np.ndarray
    t: datetime

    def __init__(self, AER, t):
        self.AER = AER
        self.t = t

    def __repr__(self) -> str:
        return f"[{self.AER[0]}, {self.AER[1]}, {self.AER[2]}], {self.t}"

class Frame():
    def __init__(self, data, az, tStart, settings):
        self.magData = np.absolute(data)
        self.az = az
        self.tStart = tStart
        self.settings = settings

    def getReturns(self) -> List[Return]:
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
            print(f"r = {r}, el = {el}")
            if r < minRange or r > maxRange:
                continue
            targetLLA = pymap3d.aer2geodetic(self.az, el, r, *self.settings['transmitterLLA'])
            returns.append(Return(targetLLA, self.tStart))
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
        

class Scan():
    magData: np.ndarray
    settings: dict # Not an int...
    tStart: float
    targetTimes = List[float]

    def __init__(self, data, settings, tStart):
        self.magData = np.absolute(data)
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

    def getFramesTimesCorrected(self) -> np.ndarray:
        centers, threshold = self.getFramesTimes()
        centers = centers * self.settings['sampleRate']
        
        # firstShortFound = False
        # idxFirstLongCPI = -1
        # for idx, interval in enumerate(np.diff(centers / self.settings['sampleRate'])):
        #     print(interval)
        #     if firstShortFound and np.isclose(interval, 1e-3, 0.01):
        #         idxFirstLongCPI = idx
        #         break
            
        #     if not firstShortFound and np.isclose(interval, 0.78e-3, 0.01):
        #         firstShortFound = True
        #         print(f"found first short!")
        
        # pulseTimes = [centers[idxFirstLongCPI]]
        # tLength = self.getLength()

        # CPISize = 10
        # longPulseLength = int(1e-3 * self.settings['sampleRate'])
        # shortPulseLength = int(0.78e-3 * self.settings['sampleRate'])

        # longCPILength = int((1e-3 * CPISize) * self.settings['sampleRate'])
        # shortCPILength = int((0.78e-3 * CPISize) * self.settings['sampleRate'])
        # while pulseTimes[-1] < len(magData):
        #     # # Load in data chunk at least one long CPI in size and see if data best matches a long or short CPI
        #     # nextCPI = magData[[pulseTimes-1] : longCPILength]
        #     # # Downsample to 10KHz
        #     # downsampleRate = int(round(self.settings['sampleRate'] / 10e3))
        #     # downsampleWindow = maxPool1d(nextCPI, downsampleRate)
        #     for _ in range(10):
        #         pulseTimes.append(pulseTimes[-1] + longPulseLength)
        #     for _ in range(10):
        #         pulseTimes.append(pulseTimes[-1] + shortPulseLength)

        # print(pulseTimes)
        

        # Correct pulse times about what we know about the transmitter
        shortPulse = int(0.78e-3 * self.settings['sampleRate'])
        longPulse = int(1e-3 * self.settings['sampleRate'])
        pulseIdxs = [centers[0]]
        for idx, c in enumerate(centers[1:]):
            interval = c - pulseIdxs[-1]
            if interval < shortPulse*0.9:
                continue # Not a valid pulse
            elif interval > shortPulse*0.9 and interval < shortPulse*1.1:
                pulseIdxs.append(pulseIdxs[-1] + shortPulse)
            elif interval > longPulse*0.95 and interval < longPulse*1.1:
                pulseIdxs.append(pulseIdxs[-1] + longPulse)
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
        # pulseTimes = centers / self.settings['sampleRate'] # For uncorrected
        # pulseTimes = pulseTimes - directPulseDelay
        return (pulseTimes, threshold)

    def toFrames(self, useCorrectedTimes=True) -> List[Frame]:
        # Get the start time of each frame according to pulse locations
        if useCorrectedTimes:
            pulseTimes, _ = self.getFramesTimesCorrected()
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

    def applyMatchedFilter(self):
        # Apply matched filter to self and return
        pass

    def applyPulseIntegration(self, nIntegrations=4):
        frames = self.toFrames(False)[1:]
        outData = np.array([])
        for i in range(nIntegrations):
            outData = np.append(outData, frames[i].magData)
            # print(len(frames[i].magData))

        startFrame = frames[0]
        startFrame.magData = startFrame.magData[:int(self.settings['sampleRate'] * 600e-6)]
        runningSum = sum(frames[1:nIntegrations], startFrame)
        for i in range(nIntegrations, len(frames)):
            if i % 100 == 0:
                print(f"Summing frame {i} out of {len(frames)}")
            # print(len(frames[i].magData))
            runningSum = runningSum + frames[i]
            runningSum = runningSum - frames[i-nIntegrations]
            outData = np.append(outData, runningSum.magData)

            # print(len(runningSum.magData))

            # plt.plot(runningSum.magData)
            # plt.show()
        
        return Scan(outData, self.settings, self.tStart)


    def getClutterSuppressedMag(self) -> np.ndarray:
        frameTimes, _ = self.getFramesTimes()
        magData = self.getMag()
        print(len(magData))
        # Apply matched filter
        
        # Apply pulse integration
        nIntegrations = 4
        frameIndexes = np.int64(np.round(frameTimes*self.settings['sampleRate']))
        frameIndexes = np.append([0], frameIndexes)
        frameIndexes = np.append(frameIndexes, [len(magData)])

        frameData = []
        for start, end in zip(frameIndexes[:-1], frameIndexes[1:]):
            frameData.append(magData[start:end])

        PIData = np.array([])
        for i in range(nIntegrations):
            PIData = np.append(PIData, frameData[i])

        for i in range(nIntegrations, len(frameData)):
            frameWindow = frameData[i-nIntegrations : i]
            intWindow = frameWindow[-1]
            for window in frameWindow[:-1]:
                diff = len(intWindow) - len(window)
                if diff == 0:
                    intWindow = intWindow + window
                elif diff > 0: # previous window is short than this one
                    intWindow[:len(window)] = intWindow[:len(window)] + window
                else: # Previous window is longer
                    intWindow = intWindow + window[len(intWindow)]
            
            PIData = np.append(PIData, intWindow)

        plt.plot(magData)
        plt.plot(PIData/nIntegrations)
        plt.show()
        print(len(PIData))
        # Apply MTI

        return PIData


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
