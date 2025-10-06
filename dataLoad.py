import csv
import glob
import gzip
from datetime import datetime, timedelta
from typing import List

import numpy as np

from ACState import ACPosition, ACVelocity, ACState

def loadCollect(fname):
    return (loadACState(fname), loadRF(fname))

def loadACState(fname):
    posFile = glob.glob(fname + '_pos.csv')[0]
    velFile = glob.glob(fname + '_vel.csv')[0]
    pos = loadPos(posFile)
    vel = loadVel(velFile)
    state = ACState(pos[0].icao, pos, vel)
    return state

def loadRF(fname) -> np.ndarray:
    rfFile = glob.glob(fname + '.dat*')[0]
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
