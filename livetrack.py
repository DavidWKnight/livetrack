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

IDX_GROUND_SPEED = 12
IDX_TRACK = 13
IDX_ALT_RATE = 16

class TrackedAircraft():
    def __init__(self, icao, tLast, posFile, velFile):
        self.icao = icao
        self.tLast = tLast
        self.posFile = posFile
        self.velFile = velFile

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
            trackedAircraft[icao].posFile.close()
            trackedAircraft[icao].velFile.close()
            del trackedAircraft[icao]

        # Try parsing next message
        line = socket_file.readline()
        if not line:  # Empty line indicates end of stream (connection closed)
            break
        line = line.decode('utf-8')
        line = line.split(',')
        icao = line[IDX_ICAO]
        if line[1] == "4":
            if icao in trackedAircraft:
                trackedAircraft[icao].velFile.write(icao)
                trackedAircraft[icao].velFile.write(", " + now.isoformat())
                trackedAircraft[icao].velFile.write(", " + line[IDX_GROUND_SPEED])
                trackedAircraft[icao].velFile.write(", " + line[IDX_TRACK])
                trackedAircraft[icao].velFile.write(", " + line[IDX_ALT_RATE])
                trackedAircraft[icao].velFile.write("\n")
            continue
        if line[1] != "3":
            continue
        if line[IDX_LAT] == '':
            continue
        if line[IDX_LON] == '':
            continue
        if line[IDX_ALT] == '':
            continue

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
                
                positionLog = open(fname + '_pos.txt', 'w+')
                velocityLog = open(fname + '_vel.txt', 'w+')
                trackedAircraft[icao] = TrackedAircraft(icao, now, positionLog, velocityLog)
                positionLog.write("ICAO, Time, Lat, Lon, Alt\n")
                velocityLog.write("ICAO, Time, Ground Speed, Track, Alt Rate\n")
                print(f"{icao} entered")
            
            trackedAircraft[icao].tLast = now
            trackedAircraft[icao].posFile.write(icao)
            trackedAircraft[icao].posFile.write(", " + now.isoformat())
            trackedAircraft[icao].posFile.write(", " + str(aircraftLLA[0]))
            trackedAircraft[icao].posFile.write(", " + str(aircraftLLA[1]))
            trackedAircraft[icao].posFile.write(", " + str(aircraftLLA[2]))
            trackedAircraft[icao].posFile.write("\n")

finally:
    pluto.stop()

    # Close the file-like object and the socket
    socket_file.close()
    sock.close()
