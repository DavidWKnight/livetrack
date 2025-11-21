import numpy as np
import pymap3d

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise, Saver

from RadReturn import RadReturn

class PlaneKF():
    """Simple Kalman Filter that tracks an aircraft in the ENU frame"""
    def __init__(self, centerLLA, pos, vel, t, P):
        self.centerLLA = np.array(centerLLA)

        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])
        self.kf.P = P

        self.t = [t]

        self.s = Saver(self.kf)

    def addReturn(self, ret: RadReturn):
        dt = (ret.t - self.t[-1]).seconds

        # State transition Matrix
        F = np.eye(6)
        F[0,1] = dt
        F[2,3] = dt
        F[4,5] = dt
        # Process Noise
        Q = Q_discrete_white_noise(2, dt=dt, var=20, block_size=3)

        self.kf.predict(F=F, Q=Q)

        # State transition matrix
        H = np.array([
            [1., 0, 0, 0, 0, 0],
            [ 0, 0, 1, 0, 0, 0],
            [ 0, 0, 0, 0, 1, 0]
        ])

        R = ret.getCov()
        estimate = pymap3d.geodetic2enu(*ret.LLA, *self.centerLLA)
        
        self.kf.update(estimate, R=R, H=H)
        self.s.save()

        self.t.append(ret.t)

    def getPos(self) -> np.ndarray:
        return self.kf.x[::2]

    def getVel(self) -> np.ndarray:
        return self.kf.x[1::2]

    def getSaver(self) -> Saver:
        self.s.to_array()
        return self.s
