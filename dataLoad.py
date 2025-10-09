import csv
import glob
import gzip
import json
from datetime import datetime, timedelta
from typing import List, BinaryIO

import numpy as np
import matplotlib.pyplot as plt

from radarAlgo import findDirectPathPulses, findPulseTime
from RFTypes import Scan
from ACState import ACPosition, ACVelocity, ACState

def loadCollect(fname):
    acStart = loadACState(fname)
    settings = loadSettings(fname)
    rfData = RFDataManager(fname, settings)
    return (acStart, settings, rfData)

def loadACState(fname):
    posFile = glob.glob(fname + '_pos.csv', recursive=True)[0]
    velFile = glob.glob(fname + '_vel.csv', recursive=True)[0]
    pos = loadPos(posFile)
    vel = loadVel(velFile)
    state = ACState(pos[0].icao, pos, vel)
    return state

def loadRF(fname) -> np.ndarray:
    rfFile = glob.glob(fname + '.dat*', recursive=True)[0]
    if rfFile.endswith('.gz'):
        with gzip.open(rfFile, 'rb') as file:
            a = file.read()
            data = np.frombuffer(a, np.int16)
            data = data[::2] + 1j * data[1::2]
            return data
    elif rfFile.endswith('.dat'):
        data = np.fromfile(rfFile)
        return data
    return None

def loadPos(fname) -> List[ACPosition]:
    with open(fname, 'r') as file:
        csv_reader = csv.reader(file)
        posData = list(csv_reader)
        
        def removeLineSpaces(a):
            return list(map(lambda x: x.lstrip(), a))
        
        header = removeLineSpaces(posData[0])
        icaoIdx = header.index('ICAO')
        tIdx = header.index('t')
        latIdx = header.index('Lat')
        lonIdx = header.index('Lon')
        altIdx = header.index('Alt')

        posData = posData[1:] # Remove header
        pos = []
        for line in posData:
            line = removeLineSpaces(line)
            icao = line[icaoIdx]
            t = datetime.fromisoformat(line[tIdx]) - timedelta(seconds=0.1)
            lla = np.array([float(line[latIdx]), float(line[lonIdx]), float(line[altIdx])])
            pos.append(ACPosition(icao, t, lla))
        return pos
    return None

def loadVel(fname) -> List[ACVelocity]:
    return []

def loadSettings(fname) -> dict:
    settingsFile = glob.glob(fname + '_settings.json', recursive=True)[0]
    with open(settingsFile, 'r') as inFile:
        settings = json.load(inFile)
    
    settings['tStart'] = datetime.fromisoformat(settings['tStart'])
    return settings

class RFDataManager():
    settings: dict
    handle: BinaryIO
    dataBuffer: np.ndarray
    foundScanBoundary: bool
    tEnd: float # Number of seconds since start of collect in the end of the dataBuffer array
    endOfFile: bool

    def __init__(self, fname: str, settings):
        self.settings = settings

        rfFile = glob.glob(fname + '.dat*', recursive=True)[0]
        if rfFile.endswith('.gz'):
            self.handle = gzip.open(rfFile, 'rb')
        elif rfFile.endswith('.dat'):
            self.handle = open(rfFile, 'rb')
        
        self.dataBuffer = np.array([])
        self.foundScanBoundary = False
        self.tEnd = 0.0
        self.endOfFile = False
    
    def load(self, nSamples: int) -> np.ndarray:
        bytesPerSample = 4
        data = self.handle.read(int(bytesPerSample*nSamples))
        data = np.frombuffer(data, np.int16)
        data = data[::2] + 1j * data[1::2]
        return data

    def loadBufferToSize(self, nSamples: int):
        loadSize = int(nSamples) - len(self.dataBuffer)
        if loadSize > 0:
            self.dataBuffer = np.append(self.dataBuffer, self.load(loadSize))
            if len(self.dataBuffer) != nSamples:
                self.endOfFile = True

            self.tEnd += loadSize / self.settings['sampleRate']
    
    def getBufferSizeSeconds(self) -> float:
        return len(self.dataBuffer) / self.settings['sampleRate']

    def acquireScanBoundary(self) -> int:
        # Can iteratively try larger buffer sizes until we acquire
        self.loadBufferToSize(int(15 * self.settings['sampleRate']))
        buffer = np.absolute(self.dataBuffer)
        # Could try match filtering self.dataBuffer to improve chances of aquiring boundary
        _, directPathIdx = findDirectPathPulses(buffer, self.settings['sampleRate'], 1, False)
        self.foundScanBoundary = True
        return directPathIdx[0]

    def getNextScan(self) -> Scan:
        if not self.foundScanBoundary:
            scanStartIdx = self.acquireScanBoundary()
            self.dataBuffer = self.dataBuffer[scanStartIdx:] # Skip garbage from before first direct pulse

        directWindowCenter = int(4.6 * self.settings['sampleRate'])
        directWindowSize = int(0.2 * self.settings['sampleRate'])
        dwStart = directWindowCenter - directWindowSize
        dwEnd = directWindowCenter + directWindowSize
        self.loadBufferToSize(dwEnd)
        directWindow = np.absolute(self.dataBuffer[dwStart:dwEnd])
        # Could try match filtering directWindow to get a better estimate
        _, nextBoundaryIdx = findPulseTime(directWindow, self.settings['sampleRate'])
        nextBoundaryIdx = nextBoundaryIdx + dwStart
        # TODO: If we can't find a viable pulse time we need to call self.acquireScanBoundary()

        scanData = self.dataBuffer[:nextBoundaryIdx]
        self.dataBuffer = self.dataBuffer[nextBoundaryIdx:]

        tStart = self.tEnd - self.getBufferSizeSeconds()
        return Scan(scanData, self.settings, tStart)

    def isEndOfFile(self) -> bool:
        return self.endOfFile
