import time
import socket
from datetime import datetime

import numpy as np
import pymap3d

from PlutoLogger import PlutoLogger

outputFolder = "collects/"

# Create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to a server (replace with your server's address and port)
server_address = ('localhost', 30003)
sock.connect(server_address)

socket_file = sock.makefile('rb')

# transmitterLLA = np.array([34.052724, -117.596634, 282])
transmitterLLA = np.array([34.1334345,-117.9070175, 198])

transmitterEl = [40, 60] # Testing values for now

IDX_ICAO = 4
IDX_CALLSIGN = 5
IDX_ALT = 11
IDX_LAT = 14
IDX_LON = 15

class TrackedAircraft():
    def __init__(self, icao, tLast, fileHandle):
        self.icao = icao
        self.tLast = tLast
        self.fileHandle = fileHandle

trackedAircraft = {} # icao : tLast
aicraftTimeout = 15

pluto = PlutoLogger()
pluto.start()

tLastSleep = datetime.now()

try:
    while True:
        # Save some cpu
        if (datetime.now() - tLastSleep).total_seconds() > 0.25:
            time.sleep(0.25)
            pass

        # Prune old entries
        aicraftToRemove = []
        for icao in trackedAircraft:
            if (datetime.now() - trackedAircraft[icao].tLast).total_seconds() > aicraftTimeout:
                aicraftToRemove.append(icao)

        for icao in aicraftToRemove:
            print(f"{icao} exited")
            pluto.stopRecording(icao)
            trackedAircraft[icao].fileHandle.close()
            del trackedAircraft[icao]

        # Try parsing next message
        line = socket_file.readline()
        if not line:  # Empty line indicates end of stream (connection closed)
            break
        line = line.decode('utf-8')
        if "MSG,3" not in line:
            continue
        line = line.split(',')
        if line[IDX_LAT] == '':
            continue
        if line[IDX_LON] == '':
            continue
        if line[IDX_ALT] == '':
            continue
        icao = line[IDX_ICAO]
        lat = float(line[IDX_LAT])
        lon = float(line[IDX_LON])
        alt = float(line[IDX_ALT])

        if np.isclose(lat, 0.0):
            continue
        if np.isclose(lon, 0.0):
            continue
        if np.isclose(alt, 0.0):
            continue
        now = datetime.now()

        aircraftLLA = np.array([lat, lon, alt])

        aircraftAER = pymap3d.geodetic2aer(*aircraftLLA, *transmitterLLA)
        if aircraftAER[1] > transmitterEl[0] and aircraftAER[1] < transmitterEl[1]:
            if icao not in trackedAircraft:
                fname = outputFolder + now.isoformat(timespec='seconds') + "_" + icao
                pluto.startRecording(icao, fname + '.dat')
                
                aircraftLog = open(fname + '.txt', 'w+')
                trackedAircraft[icao] = TrackedAircraft(icao, now, aircraftLog)
                aircraftLog.write("ICAO, Time, Lat, Lon, Alt\n")
                print(f"{icao} entered")
            
            trackedAircraft[icao].tLast = now
            trackedAircraft[icao].fileHandle.write(icao)
            trackedAircraft[icao].fileHandle.write(", " + now.isoformat())
            trackedAircraft[icao].fileHandle.write(", " + str(aircraftLLA[0]))
            trackedAircraft[icao].fileHandle.write(", " + str(aircraftLLA[1]))
            trackedAircraft[icao].fileHandle.write(", " + str(aircraftLLA[2]))
            trackedAircraft[icao].fileHandle.write("\n")

finally:
    pluto.stop()

    # Close the file-like object and the socket
    socket_file.close()
    sock.close()
