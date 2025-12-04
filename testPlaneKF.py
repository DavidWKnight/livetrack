import math
from datetime import datetime, timedelta

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

import pymap3d
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver
import filterpy.stats as stats

from PlaneKF import PlaneKF
from RadReturn import RadReturn
import util

def planeCirclePath(t):
    # Circular movement no change in altitude
    freq = 1/(360)
    amp = 3e3
    freq2pi = freq*2*np.pi

    vertFreq = 1/(360)
    vertAmp = 1e3
    vertfreq2pi = vertFreq*2*np.pi

    x = amp * np.cos(t*freq2pi)
    y = amp * np.sin(t*freq2pi)
    dx = amp * -freq2pi * np.sin(t*freq2pi)
    dy = amp * freq2pi * np.cos(t*freq2pi)

    z = vertAmp * np.sin(t*vertfreq2pi)
    dz = vertAmp * -vertfreq2pi * np.sin(t*vertfreq2pi)

    return [x,y,z,dx,dy,dz]

def testPlaneKF():
    np.random.seed(0)

    tSim = 360
    tSample = 5
    
    truthTimes = np.arange(0, tSim, tSample)
    randomSampleTimes = np.random.uniform(low=0, high=360, size=50)
    truthTimes = np.sort(np.concatenate([truthTimes, randomSampleTimes]))
    measurementTimes = truthTimes[1:]

    pos = np.array([10e3,10e3,1e3]) # Meters ENU
    vel = np.array([5, -2, -3]) # Meters

    
    stateNonLinear = np.array(list(map(planeCirclePath, truthTimes)))
    stateLinear = np.tile(truthTimes, (len(vel),1)).T * vel
    pTruth = pos + stateNonLinear[:,:3] + stateLinear

    # Initial state error
    posInit = pTruth[0] + np.array([500, -300, 200])
    velInit = pTruth[1] + np.array([-30, 10, 20])

    P = np.diag([10e3, 50, 10e3, 50, 5e3, 25])

    centerLLA = [34.052724, -117.596634, 0]
    tStart = datetime(2025, 1, 1, 0, 0, 0)
    pkf = PlaneKF(centerLLA, posInit, velInit, tStart, P)


    aerStd = np.array([util.ASR9_AZ_STD, util.ASR9_EL_STD, util.ASR9_RANGE_STD])
    
    
    error = [np.linalg.norm(pkf.getPos() - pTruth[0])]
    estimateError = []
    xyError = [np.linalg.norm(pkf.getPos()[:2] - pTruth[0,:2])]
    elevationError = [pkf.getPos()[2] - pTruth[0,2]]

    for tIdx in range(len(measurementTimes)):
        t = tStart + timedelta(seconds=int(measurementTimes[tIdx]))
        # Because motion is linear we can use a simple propagation model
        truthAER = np.array(pymap3d.enu2aer(pTruth[tIdx][0], pTruth[tIdx][1], pTruth[tIdx][2]))
        estimateAER = truthAER + (np.random.randn(1, 3)[0] * aerStd)
        estimateLLA = pymap3d.aer2geodetic(*estimateAER, *centerLLA)
        estimateENU = np.array(pymap3d.aer2enu(*estimateAER))
        estimateError.append(np.linalg.norm(estimateENU - pTruth[tIdx]))

        estimate = RadReturn(estimateLLA, estimateAER, t)
        pkf.estimate(estimate)

        error.append(np.linalg.norm(pkf.getPos() - pTruth[tIdx]))
        xyError.append(np.linalg.norm(pkf.getPos()[:2] - pTruth[tIdx,:2]))
        elevationError.append(pkf.getPos()[2] - pTruth[tIdx,2])

    xyMapPlot = plt.subplot2grid((3,2), (0, 0))
    elevationMap = plt.subplot2grid((3,2), (0, 1))
    xyErrorPlot = plt.subplot2grid((3,2), (1, 0))
    elevationErrorPlot = plt.subplot2grid((3,2), (1, 1))
    errorPlot = plt.subplot2grid((3,2), (2, 0), colspan=2)
    
    s = pkf.getSaver()
    xyMapPlot.scatter(s.z[:,0], s.z[:,1], label='Measurements') # type: ignore
    xyMapPlot.plot(s.x[:,0], s.x[:,2], label='Estimates') # type: ignore
    xyMapPlot.plot(pTruth[:,0], pTruth[:,1], label='Truth')
    xyMapPlot.scatter([0],[0], label='Transmitter')
    xyMapPlot.grid(True)
    xyMapPlot.legend()
    xyMapPlot.set_title('XY Position Map Plot')
    xyMapPlot.set_xlabel('Position X (m)')
    xyMapPlot.set_ylabel('Position Y (m)')

    xyErrorPlot.plot(truthTimes, xyError)
    xyErrorPlot.grid(True)
    xyErrorPlot.set_title('XY position Error')
    xyErrorPlot.set_xlabel('Time (s)')
    xyErrorPlot.set_ylabel('Error (m)')

    elevationMap.scatter(measurementTimes, s.z[:,2], label='Measurements') # type: ignore
    elevationMap.plot(measurementTimes, s.x[:,4], label='Estimates') # type: ignore
    elevationMap.plot(measurementTimes, pTruth[1:,2], label='Truth')
    elevationMap.grid(True)
    elevationMap.legend()
    elevationMap.set_title('Magnitude Error')
    elevationMap.set_xlabel('Time (s)')
    elevationMap.set_ylabel('Error (m)')

    elevationErrorPlot.plot(truthTimes, elevationError)
    elevationErrorPlot.grid(True)
    elevationErrorPlot.set_title('Elevation Error')
    elevationErrorPlot.set_xlabel('Time (s)')
    elevationErrorPlot.set_ylabel('Error (m)')

    errorPlot.plot(truthTimes, error)
    errorPlot.grid(True)
    errorPlot.set_title('Total Error')
    errorPlot.set_xlabel('Time (s)')
    errorPlot.set_ylabel('Error (m)')

    plt.tight_layout(h_pad=0)
    plt.show()
    plt.show()

if __name__ == "__main__":

    testPlaneKF()
