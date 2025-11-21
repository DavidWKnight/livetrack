from datetime import datetime

import numpy as np

import util

class RadReturn():
    LLA: np.ndarray
    AER: np.ndarray
    t: datetime

    def __init__(self, LLA, AER, t):
        self.LLA = LLA
        self.AER = AER
        self.t = t

    def getCov(self) -> np.ndarray:
        # Adapted from Principals of Radar Target Tracking Eq 19,20,21
        az = np.deg2rad(self.AER[0])
        el = np.deg2rad(self.AER[1])
        R = self.AER[2]
        azVar = util.ASR9_AZ_STD**2
        elVar = util.ASR9_EL_STD**2
        rangeVar = util.ASR9_RANGE_STD**2

        cosBearing2 = np.cos(az)**2
        sinBearing2 = np.sin(az)**2
        tanElev2 = np.tan(el)**2

        vary = rangeVar*cosBearing2 + (R**1.7)*sinBearing2*(azVar/2)
        varx = rangeVar*sinBearing2 + (R**1.7)*cosBearing2*(azVar/2)
        varz = rangeVar*tanElev2 + (R**1.7)*tanElev2*(elVar/2)
        varxy = 0.5*np.sin(2*az)*(rangeVar - (R**1.7)*(azVar/2))
        R = np.array([
            [ varx, varxy,    0],
            [varxy,  vary,    0],
            [     0,     0, varz],
        ])

        return R

    def __repr__(self) -> str:
        return f"[{self.LLA[0]}, {self.LLA[1]}, {self.LLA[2]}], {self.t}"
