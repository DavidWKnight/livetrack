import time
import json
import socket
from datetime import datetime

import numpy as np
import pymap3d

from PlutoLogger import PlutoLogger
from ACState import ACPosition, ACVelocity

outputFolder = "collects/"

# Create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to a server (replace with your server's address and port)
server_address = ('localhost', 30003)
sock.connect(server_address)

socket_file = sock.makefile('rb')

SETTINGS = {
    # "transmitterLLA": [34.052724, -117.596634, 0],
    "transmitterLLA": [34.1334345,-117.9070175, 198],
    "receiverLLA": [34.051555, -117.593415, 0],
    "sampleRate": 3e6, # Hz
    "centerFreq": 2.897028e9, # Hz
    "numSamples": 2**20,
    "bandwidth": 0.5e6,
    "rfDtype": 'float32'
}

# transmitterLLA = np.array([34.052724, -117.596634, 0])
transmitterLLA = np.array(np.array(SETTINGS['transmitterLLA']))

transmitterEl = [40, 60] # Testing values for now
print("Using fake enter/exit elavations, fix before real collects!")

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

class TrackedAircraft():
    def __init__(self, icao, tLast, posFile, velFile):
        self.icao = icao
        self.tLast = tLast
        self.posFile = posFile
        self.velFile = velFile

trackedAircraft = {} # icao : tLast
aicraftTimeout = 15

pluto = PlutoLogger(SETTINGS["sampleRate"], SETTINGS["centerFreq"], SETTINGS["numSamples"], SETTINGS["bandwidth"], SETTINGS["rfDtype"])
pluto.start()

tLastSleep = datetime.now()

allPosFile = open(outputFolder + "allPos.csv", "a+")
allVelFile = open(outputFolder + "allVel.csv", "a+")

try:
    while True:
        # Save some cpu
        if (datetime.now() - tLastSleep).total_seconds() > 0.25:
            time.sleep(0.25)
            allPosFile.flush()
            allVelFile.flush()

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
        now = datetime.now()
        if not line:  # Empty line indicates end of stream (connection closed)
            break
        line = line.decode('utf-8')
        line = line.split(',')

        # Handle velocity message
        if line[IDX_MSG_TYPE] == SBS_VELOCITY_MESSAGE:
            msg = ACVelocity().fromSBS(line, now)
            allVelFile.write(msg.toCSVLine())
            if msg.icao in trackedAircraft:
                trackedAircraft[msg.icao].velFile.write(msg.toCSVLine())
            continue

        if line[IDX_MSG_TYPE] != SBS_POSITION_MESSAGE:
            continue
        
        # Handle position message
        try:
            msg = ACPosition().fromSBS(line, now)
        except Exception as e:
            print(f"Bad position message! {e} - {line}")
            continue
        icao = msg.icao
        
        allPosFile.write(msg.toCSVLine())

        # Check if aircraft is in the transmitters "FOV"
        # aircraftAER = pymap3d.geodetic2aer(*msg.LLA, *transmitterLLA)
        # if aircraftAER[1] > transmitterEl[0] and aircraftAER[1] < transmitterEl[1]:

        aircraftENU = pymap3d.geodetic2enu(*msg.LLA, *transmitterLLA)
        # 20km before/after the runway, +- 3 km north/south of the runway, and <5000 feet
        if abs(aircraftENU[0]) < 20e3 and abs(aircraftENU[1]) < 6e3 and aircraftENU[2] < 5000*0.3048:
            # Check if this is the first time we're seeing this aircraft
            if icao not in trackedAircraft:
                fname = outputFolder + now.isoformat(timespec='seconds') + "_" + icao
                pluto.startRecording(icao, fname + '.dat')
                
                with open(fname + "_settings.json", 'w+') as stateFile:
                    state = SETTINGS
                    state['tStart'] = now.isoformat(timespec='seconds')
                    json.dump(state, stateFile)

                positionLog = open(fname + '_pos.csv', 'w+')
                positionLog.write(ACPosition().getCSVHeader())

                velocityLog = open(fname + '_vel.csv', 'w+')
                velocityLog.write(ACVelocity().getCSVHeader())

                trackedAircraft[icao] = TrackedAircraft(icao, now, positionLog, velocityLog)
                print(f"{icao} entered")
            
            trackedAircraft[icao].tLast = now
            trackedAircraft[icao].posFile.write(msg.toCSVLine())
        else:
            print(f"Ignoring {icao}, position: {aircraftENU}")

finally:
    pluto.stop()

    allPosFile.close()
    allVelFile.close()

    for aircraft in trackedAircraft.values():
        aircraft.posFile.close()
        aircraft.velFile.close()

    # Close the file-like object and the socket
    socket_file.close()
    sock.close()
