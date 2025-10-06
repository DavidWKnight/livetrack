from PlutoLogger import PlutoLogger

import time

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



pluto = PlutoLogger(SETTINGS["sampleRate"], SETTINGS["centerFreq"], SETTINGS["numSamples"], SETTINGS["bandwidth"], SETTINGS["rfDtype"])
pluto.start()

pluto.startRecording('AAAAAA', 'collect' + '.dat')
# pluto.startRecording('BBBBBB', 'collect2' + '.dat')
time.sleep(15)
pluto.stopRecording('AAAAAA')
# pluto.stopRecording('BBBBBB')

pluto.stop()

