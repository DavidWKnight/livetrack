from datetime import datetime
from typing import List

import numpy as np
import pymap3d

IDX_MSG_TYPE = 1
IDX_ICAO = 4
IDX_CALLSIGN = 5
IDX_ALT = 11
IDX_LAT = 14
IDX_LON = 15

IDX_GROUND_SPEED = 12
IDX_TRACK = 13
IDX_ALT_RATE = 16

SBS_POSITION_MESSAGE = "3"
SBS_VELOCITY_MESSAGE = "4"

def lerp(a, b, t):
  """
  Performs linear interpolation between two values.

  Args:
    a (float): The starting value.
    b (float): The ending value.
    t (float): The interpolation amount, typically between 0.0 and 1.0.

  Returns:
    float: The interpolated value.
  """
  return a + (b - a) * t

class ACPosition:
    icao: str
    t: datetime
    LLA: np.array # Altitude in meters

    def __init__(self, icao="", t=datetime(2000, 1, 1, 0, 0, 0), LLA=np.array([0.0, 0.0, 0.0])):
        self.icao = icao
        self.t = t
        self.LLA = LLA

    def fromSBS(self, line, t):
        if line[IDX_MSG_TYPE] != SBS_POSITION_MESSAGE:
            raise ValueError("Not a position message!")
        if line[IDX_LAT] == '':
            raise ValueError("Bad lat " + line[IDX_LAT])
        if line[IDX_LON] == '':
            raise ValueError("Bad lon " + line[IDX_LAT])
        if line[IDX_ALT] == '':
            raise ValueError("Bad alt " + line[IDX_LAT])

        lat = float(line[IDX_LAT])
        lon = float(line[IDX_LON])
        alt = float(line[IDX_ALT]) * 0.3048

        # According to google AI thing
        localPressure = 30.05 # 9_28 @ 10:30am
        altMSL = float(line[IDX_ALT]) + (localPressure - 29.92)*1000

        if np.isclose(lat, 0.0):
            raise ValueError("Lat near 0")
        if np.isclose(lon, 0.0):
            raise ValueError("Lon near 0")
        if np.isclose(alt, 0.0):
            raise ValueError("Alt near 0")

        self.icao = line[IDX_ICAO]
        self.t = t
        self.LLA = np.array([lat, lon, alt])
        return self

    def getCSVHeader(self) -> str:
        return "ICAO, t, Lat, Lon, Alt\n"

    def toCSVLine(self) -> str:
        line = ""
        line += self.icao
        line += ", " + self.t.isoformat()
        line += ", " + str(self.LLA[0])
        line += ", " + str(self.LLA[1])
        line += ", " + str(self.LLA[2])
        line += "\n"
        return line

    def lerp(self, other, t):
        # Only valid if 

        icao = self.icao
        dt = (other.t - self.t).total_seconds()
        tLerp = (t - self.t).total_seconds()
        if dt > 60:
            return None # This is a terrible approximation

        a = np.array(pymap3d.geodetic2ecef(*self.LLA))
        b = np.array(pymap3d.geodetic2ecef(*other.LLA))
        x = tLerp / dt
        c = lerp(a, b, x)
        pos = np.array(pymap3d.ecef2geodetic(*c))
        return ACPosition(icao, t, pos)

class ACVelocity:
    icao: str
    t: datetime
    heading: float # Degrees
    groundSpeed: float # ground speed in meters/s
    altRate: float # meters/s

    def __init__(self, icao="", t=datetime(2000, 1, 1, 0, 0, 0), heading=0.0, groundSpeed=0.0, altRate=0.0):
        self.icao = icao
        self.t = t
        self.heading = heading
        self.groundSpeed = groundSpeed
        self.altRate = altRate
        
    def fromSBS(self, line, t):
        if line[IDX_MSG_TYPE] != SBS_VELOCITY_MESSAGE:
            raise ValueError("Not a velocity message!")
        
        self.icao        = line[IDX_ICAO]
        self.t           = t
        self.heading     = float(line[IDX_GROUND_SPEED])
        self.groundSpeed = float(line[IDX_TRACK])
        self.altRate     = float(line[IDX_ALT_RATE])

        return self
    
    def getCSVHeader(self) -> str:
        return "ICAO, t, Ground Speed, Track, Alt Rate\n"

    def toCSVLine(self) -> str:
        line = ""
        line += self.icao
        line += ", " + self.t.isoformat()
        line += ", " + str(self.heading)
        line += ", " + str(self.groundSpeed)
        line += ", " + str(self.altRate)
        line += "\n"
        return line

class ACState:
    icao: str
    positions = List[ACPosition]
    velocities = List[ACVelocity]

    def __init__(self, icao, positions=[], velocities=[]):
        self.icao = icao
        self.positions = positions
        self.velocities = velocities

    def getPosition(self, t: datetime, extrap: bool=False) -> np.array:
        # Find the first recording that is after the requested time
        t2Idx = -1
        for idx, pos in enumerate(self.positions):
            if t < pos.t:
                t2Idx = idx
                break

        t1Idx = t2Idx - 1
        if t2Idx < 0:
            if extrap:
                out = self.positions[-2].lerp(self.positions[-1], t)
                return out.LLA # If you really do want it
            return None # Requested time is after this collect
        if t1Idx < 0:
            if extrap:
                out = self.positions[0].lerp(self.positions[1], t)
                return out.LLA # If you really do want it
            return None # Requested time is before this collect
        
        out = self.positions[t1Idx].lerp(self.positions[t2Idx], t)
        return out.LLA

    def getVelocity(self, t: datetime) -> np.array:
        pass

