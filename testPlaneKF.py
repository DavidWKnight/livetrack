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

def getRadarCov(AER):
    # Principals of Radar Target Tracking Eq 19,20,21
    az = np.deg2rad(AER[0])
    el = np.deg2rad(AER[1])
    R = AER[2]
    azVar = util.ASR9_AZ_STD**2
    elVar = util.ASR9_EL_STD**2
    rangeVar = util.ASR9_RANGE_STD**2

    cosBearing2 = np.cos(az)**2
    sinBearing2 = np.sin(az)**2

    vary = rangeVar*cosBearing2 + (R**2)*sinBearing2*(azVar/2)
    varx = rangeVar*sinBearing2 + (R**2)*cosBearing2*(azVar/2)
    varxy = 0.5*np.sin(2*az)*(rangeVar - (R**2)*(azVar/2))
    R = np.array([
        [ varx, -varxy],
        [-varxy,  vary]
    ])

    return R

def getRadarCov3d(AER):
    # Adapted from Principals of Radar Target Tracking Eq 19,20,21
    az = np.deg2rad(AER[0])
    el = np.deg2rad(AER[1])
    R = AER[2]
    azVar = util.ASR9_AZ_STD**2
    elVar = util.ASR9_EL_STD**2
    rangeVar = util.ASR9_RANGE_STD**2

    cosBearing2 = np.cos(az)**2
    sinBearing2 = np.sin(az)**2
    tanElev2 = np.tan(el)**2

    vary = rangeVar*cosBearing2 + (R**1.5)*sinBearing2*(azVar/2)
    varx = rangeVar*sinBearing2 + (R**1.5)*cosBearing2*(azVar/2)
    varz = rangeVar*tanElev2 + (R**1.5)*tanElev2*(elVar/2)
    varxy = 0.5*np.sin(2*az)*(rangeVar - (R**1.5)*(azVar/2))
    R = np.array([
        [ varx, -varxy,    0],
        [-varxy,  vary,    0],
        [     0,     0, varz],
    ])

    return R

def test3d():
    tSim = 180
    tSample = 1
    dt = tSample

    pos = np.array([1e3, 1e3, 2e3]) # Meters ENU
    vel = np.array([100, 10, 0]) # Meters

    kf = KalmanFilter(dim_x=6, dim_z=3)

    # Initial state
    kf.x = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])

    # Initial state error
    pos = pos + np.array([200, 600, -100])
    
    # Initial state covariance
    kf.P = np.diag([10e3, 50, 10e3, 50, 10e3, 50])

    # State transition Matrix
    kf.F = np.eye(6)
    kf.F[0,1] = dt
    kf.F[2,3] = dt
    kf.F[4,5] = dt
    kf.H = np.array([
        [1., 0, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0]
    ])
    
    # Measurement uncertainty
    kf.R = np.array([
        [50.,  0,   0],
        [  0, 50,   0],
        [  0,   0, 50],
    ])
    z_std = math.sqrt(kf.R[0,0])

    # Process noise
    kf.Q = Q_discrete_white_noise(2, dt=dt, var=10, block_size=3)

    # pTruth = pos + np.stack([np.linspace(0, tSim-1, tSim), np.linspace(0, tSim-1, tSim)]).T*vel
    pTruth = pos + np.tile(np.linspace(0, tSim-1, tSim), (3,1)).T * vel

    xs, cov = [kf.x], [kf.P]
    zs = []
    error = [np.linalg.norm(kf.x[::2] - pTruth[0])]
    for t in np.arange(tSample, tSim, tSample):
        # Because motion is linear we can use a simple propagation model
        zs.append(pTruth[t] + randn() * z_std)
        kf.predict()
        kf.update(zs[-1])
        xs.append(kf.x)
        cov.append(kf.P)
        error.append(np.linalg.norm(kf.x[::2] - pTruth[t]))
    

    xs, cov = np.array(xs), np.array(cov)
    zs = np.array(zs)
    plt.plot(np.linspace(0, tSim-1, tSim), error)
    # plt.scatter(zs[:,0], zs[:,1], label='Measurements')
    # plt.plot(xs[:,0], xs[:,2], label='Estimates')
    # plt.plot(pTruth[:,0], pTruth[:,1], label='Truth')
    # plt.legend()
    # plt.show()
    plt.show()

def test2dRadar():
    np.random.seed(0)

    tSim = 360
    tSample = 1
    dt = tSample

    pos = np.array([-10e3, 10e3]) # Meters ENU
    vel = np.array([50, 0]) # Meters

    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Initial state
    kf.x = np.array([pos[0], vel[0], pos[1], vel[1]])

    pTruth = pos + np.tile(np.linspace(0, tSim-1, tSim), (len(vel),1)).T * vel

    # Initial state error
    pos = pos + np.array([50, -100])
    
    # Initial state covariance
    kf.P = np.diag([10e3, 50, 10e3, 50])

    # State transition Matrix
    kf.F = np.eye(4)
    kf.F[0,1] = dt
    kf.F[2,3] = dt
    kf.H = np.array([
        [1., 0, 0, 0],
        [ 0, 0, 1, 0]
    ])

    # Process noise
    kf.Q = Q_discrete_white_noise(2, dt=dt, var=10, block_size=2)

    s = Saver(kf)

    error = [np.linalg.norm(kf.x[::2] - pTruth[0])]
    for t in np.arange(tSample, tSim, tSample):
        # Because motion is linear we can use a simple propagation model
        R = getRadarCov(pymap3d.enu2aer(pTruth[t][0], pTruth[t][1], 0))
        
        # stats.plot_covariance(pTruth[t], R)
        estimate = pTruth[t] + randn(len(R))*np.sqrt(np.diag(R))

        kf.predict()
        kf.update(estimate, R=R)
        s.save()

        error.append(np.linalg.norm(kf.x[::2] - pTruth[t]))
    s.to_array()
    fig, (xyMapPlot, errorPlot) = plt.subplots(2,1)
    
    xyMapPlot.scatter(s.z[:,0], s.z[:,1], label='Measurements') # type: ignore
    xyMapPlot.plot(s.x[:,0], s.x[:,2], label='Estimates') # type: ignore
    xyMapPlot.plot(pTruth[:,0], pTruth[:,1], label='Truth')
    xyMapPlot.legend()
    xyMapPlot.set_title('XY Position Map Plot')
    xyMapPlot.set_xlabel('Position X (m)')
    xyMapPlot.set_ylabel('Position Y (m)')

    errorPlot.plot(np.arange(0, tSim, tSample), error)
    errorPlot.set_title('Magnitude Error')
    errorPlot.set_xlabel('Time (s)')
    errorPlot.set_ylabel('Error (m)')
    plt.show()
    plt.show()

def test3dRadar():
    np.random.seed(0)

    tSim = 360
    tSample = 5
    dt = tSample

    pos = np.array([-10e3, 10e3, 5e3]) # Meters ENU
    vel = np.array([50, -20, -30]) # Meters

    # pTruth = pos + np.tile(np.linspace(0, tSim-1, tSim), (len(vel),1)).T * vel
    pTruth = pos + np.tile(np.arange(0, tSim, tSample), (len(vel),1)).T * vel

    kf = KalmanFilter(dim_x=6, dim_z=3)

    # Initial state error
    posInit = pos + np.array([500, -300, 200])
    velInit = vel + np.array([-30, 10, 20])

    # Initial state
    kf.x = np.array([posInit[0], velInit[0], posInit[1], velInit[1], posInit[2], velInit[2]])
    
    # Initial state covariance
    kf.P = np.diag([10e3, 50, 10e3, 50, 5e3, 25])

    # State transition Matrix
    kf.F = np.eye(6)
    kf.F[0,1] = dt
    kf.F[2,3] = dt
    kf.F[4,5] = dt
    kf.H = np.array([
        [1., 0, 0, 0, 0, 0],
        [ 0, 0, 1, 0, 0, 0],
        [ 0, 0, 0, 0, 1, 0]
    ])

    # Process noise
    kf.Q = Q_discrete_white_noise(2, dt=dt, var=20, block_size=3)

    s = Saver(kf)

    aerStd = np.array([util.ASR9_AZ_STD, util.ASR9_EL_STD, util.ASR9_RANGE_STD])
    
    measurementTimes = np.arange(tSample, tSim, tSample)
    error = [np.linalg.norm(kf.x[::2] - pTruth[0])]
    xyError = [np.linalg.norm(kf.x[:4:2] - pTruth[0,:2])]
    elevationError = [kf.x[4] - pTruth[0,2]]
    for tIdx in range(len(measurementTimes)):
        t = measurementTimes[tIdx]
        # Because motion is linear we can use a simple propagation model
        truthAER = np.array(pymap3d.enu2aer(pTruth[tIdx][0], pTruth[tIdx][1], pTruth[tIdx][2]))
        
        estimateAER = truthAER + (np.random.randn(1, 3)[0] * aerStd)
        # stats.plot_covariance(pTruth[t], R)
        estimate = np.array(pymap3d.aer2enu(*estimateAER))
        print(f"estimate = {estimate}")
        R = getRadarCov3d(estimateAER)
        kf.predict()
        kf.update(estimate, R=R)
        s.save()

        error.append(np.linalg.norm(kf.x[::2] - pTruth[tIdx]))
        xyError.append(np.linalg.norm(kf.x[:4:2] - pTruth[tIdx,:2]))
        elevationError.append(kf.x[4] - pTruth[tIdx,2])
    s.to_array()

    xyMapPlot = plt.subplot2grid((3,2), (0, 0))
    elevationMap = plt.subplot2grid((3,2), (0, 1))
    xyErrorPlot = plt.subplot2grid((3,2), (1, 0))
    elevationErrorPlot = plt.subplot2grid((3,2), (1, 1))
    errorPlot = plt.subplot2grid((3,2), (2, 0), colspan=2)
    
    xyMapPlot.scatter(s.z[:,0], s.z[:,1], label='Measurements') # type: ignore
    xyMapPlot.plot(s.x[:,0], s.x[:,2], label='Estimates') # type: ignore
    xyMapPlot.plot(pTruth[:,0], pTruth[:,1], label='Truth')
    xyMapPlot.legend()
    xyMapPlot.set_title('XY Position Map Plot')
    xyMapPlot.set_xlabel('Position X (m)')
    xyMapPlot.set_ylabel('Position Y (m)')

    xyErrorPlot.plot(np.arange(0, tSim, tSample), xyError)
    xyErrorPlot.grid(True)
    xyErrorPlot.set_title('XY position Error')
    xyErrorPlot.set_xlabel('Time (s)')
    xyErrorPlot.set_ylabel('Error (m)')

    elevationMap.plot(np.arange(tSample, tSim, tSample), s.z[:,2], label='Measurements') # type: ignore
    elevationMap.plot(np.arange(tSample, tSim, tSample), s.x[:,4], label='Estimates') # type: ignore
    elevationMap.plot(np.arange(tSample, tSim, tSample), pTruth[1:,2], label='Truth')
    elevationMap.legend()
    elevationMap.set_title('Magnitude Error')
    elevationMap.set_xlabel('Time (s)')
    elevationMap.set_ylabel('Error (m)')

    elevationErrorPlot.plot(np.arange(0, tSim, tSample), elevationError)
    elevationErrorPlot.grid(True)
    elevationErrorPlot.set_title('Elevation Error')
    elevationErrorPlot.set_xlabel('Time (s)')
    elevationErrorPlot.set_ylabel('Error (m)')

    errorPlot.plot(np.arange(0, tSim, tSample), error)
    errorPlot.grid(True)
    errorPlot.set_title('Total Error')
    errorPlot.set_xlabel('Time (s)')
    errorPlot.set_ylabel('Error (m)')

    plt.tight_layout(h_pad=0)
    plt.show(block=False)

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
        R = getRadarCov3d(truthAER)
        estimateAER = truthAER + (np.random.randn(1, 3)[0] * aerStd)
        estimateLLA = pymap3d.aer2geodetic(*estimateAER, *centerLLA)
        estimateENU = np.array(pymap3d.aer2enu(*estimateAER))
        estimateError.append(np.linalg.norm(estimateENU - pTruth[tIdx]))

        estimate = RadReturn(estimateLLA, estimateAER, t)
        pkf.addReturn(estimate)

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
    # test3dRadar()
    # plt.figure()
    testPlaneKF()
    plt.show()
    # test2dRadar()
    # test3d()
