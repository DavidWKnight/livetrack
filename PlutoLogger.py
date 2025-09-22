import time
import gzip
import subprocess
from datetime import datetime
from multiprocessing import Process, Queue, Array, Event

import matplotlib.pyplot as plt
import adi

CENTER_FREQ_IDX = 0
SAMPLE_RATE_IDX = 1
BANDWIDTH_IDX = 2
NUM_SAMPLES_IDX = 3
NUM_IDXS = 4

outputFolder = "collects/"

def closeFile(handle):
    fname = handle.name
    handle.close()
    start = time.time()
    subprocess.Popen(['gzip', fname, '&&', 'rm', fname])
    end = time.time()


def run(sdrSettings, updatedSettings, runFlag, inputQueue):
    recordFiles = {}

    sdr = adi.Pluto('ip:192.168.2.1')
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 60.0 # dB

    while (runFlag.is_set()):
        if updatedSettings.is_set():
            sdr.rx_lo = int(sdrSettings[CENTER_FREQ_IDX])
            sdr.sample_rate = int(sdrSettings[SAMPLE_RATE_IDX])
            sdr.rx_rf_bandwidth = int(sdrSettings[BANDWIDTH_IDX])
            sdr.rx_buffer_size = int(sdrSettings[NUM_SAMPLES_IDX])
            updatedSettings.clear()

        while not inputQueue.empty():
            nextCommand = inputQueue.get()
            print(nextCommand)
            if nextCommand['onoff'] == 'on':
                recordFiles[nextCommand['icao']] = open(outputFolder + nextCommand['fname'], 'wb')
            else:
                if nextCommand['icao'] not in recordFiles:
                    continue
                closeFile(recordFiles[nextCommand['icao']])
                del recordFiles[nextCommand['icao']]

        if len(recordFiles) == 0:
            time.sleep(0.25) # Nothing to record, we can sleep
        else:
            start = time.time()
            frame = sdr.rx()
            for handle in recordFiles.values():
                handle.write(frame.tobytes())
            end = time.time()
    # Close all archives
    for handle in recordFiles.values():
        closeFile(handle)


class PlutoLogger():
    def __init__(self):
        self.recordingICAOs = []

        self.sdrSettings = Array('f', range(NUM_IDXS))
        self.sdrSettings[SAMPLE_RATE_IDX] = 1e6 # Hz
        self.sdrSettings[CENTER_FREQ_IDX] = 2.897028e9 # Hz
        self.sdrSettings[NUM_SAMPLES_IDX] = 4096 * 8
        self.sdrSettings[BANDWIDTH_IDX] = 2e6
        
        self.updatedSettings = Event()
        self.runFlag = Event()
        self.inputQueue = Queue()

        self.process = Process(target=run, args=(self.sdrSettings, self.updatedSettings, self.runFlag, self.inputQueue))

    def start(self):
        self.updatedSettings.set()
        self.runFlag.set()
        self.process.start()

    def stop(self):
        self.runFlag.clear()
        self.process.join()

    def startRecording(self, icao, fname):
        self.recordingICAOs.append(icao)
        self.inputQueue.put({'onoff': 'on', 'icao': icao, 'fname': fname})
    
    def stopRecording(self, icao):
        try:
            self.recordingICAOs.remove(icao)
        except:
            pass
        self.inputQueue.put({'onoff': 'off', 'icao': icao, 'fname': ''})

    def currentProcessing(self):
        # Return a list of icao's that are currently recording
        return self.recordingICAOs

    def updateSDRSettings(self):
        # Might want separate methods for each setting to update
        pass
