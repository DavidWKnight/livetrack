import numpy as np
import matplotlib.pyplot as plt

from RFTypes import Scan

def lin2db(x):
    return 20 * np.log10(x + np.finfo(float).eps)

def amplitudeSpectrogramPlot(scan: Scan, settings, show=True):
    fig, axes = plt.subplots(2, 1, sharex=True)

    timeSeriesPlot = axes[0]
    spectrogramPlot = axes[1]

    data = lin2db(scan.getMag())
    data = np.clip(data, np.median(data), np.inf) # Clip bottom to the noise floor
    timeSeriesPlot.plot(scan.getDataTimes() - scan.tStart, data)
    for tDet in scan.targetTimes:
        timeSeriesPlot.axvline(tDet - scan.tStart, color='r', linestyle='--')
    
    timeSeriesPlot.set_title('Time vs Amplitude')
    timeSeriesPlot.set_xlabel('Time (seconds)')
    timeSeriesPlot.set_ylabel('Amplitude (dB)')

    fftSize = 1024
    spectrogramPlot.specgram(scan.getMag(), Fs=settings['sampleRate'], NFFT=fftSize, noverlap=int(fftSize/2))
    spectrogramPlot.set_xlabel('Time (seconds)')
    spectrogramPlot.set_ylabel('Frequency (Hz)')
    if show:
        plt.show()

def amplitudePlot(scan: Scan, settings, show=True):
    data = lin2db(scan.getMag())
    data = np.clip(data, np.median(data), np.inf) # Clip bottom to the noise floor
    plt.plot(scan.getDataTimes() - scan.tStart, data)
    for tDet in scan.targetTimes:
        plt.axvline(tDet - scan.tStart, color='r', linestyle='--')
    
    plt.title('Time vs Amplitude')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (dB)')
    if show:
        plt.show()
