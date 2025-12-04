import gc
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import util
from Scan import Scan
from RadReturn import RadReturn

def amplitudeSpectrogramPlot(scan: Scan, settings, show=True):
    fig, axes = plt.subplots(2, 1, sharex=True)

    timeSeriesPlot = axes[0]
    spectrogramPlot = axes[1]

    data = util.lin2db(scan.getMag())
    # data = util.lin2db(scan.getClutterSuppressedMag())
    
    data = np.clip(data, np.median(data), np.inf) # Clip bottom to the noise floor
    timeSeriesPlot.plot(scan.getDataTimes() - scan.tStart, data)
    for tDet in scan.targetTimes:
        timeSeriesPlot.axvline(tDet - scan.tStart, color='r', linestyle='--')
    
    # pulseLines = np.arange(0, scan.getLength(), 1e-3)
    pulseLines, threshold = scan.getFramesTimesCorrected()
    for line in pulseLines:
        timeSeriesPlot.axvline(line, color='gray', linestyle='--')
    timeSeriesPlot.axhline(util.lin2db(threshold), color='blue')

    timeSeriesPlot.set_title('Time vs Amplitude')
    timeSeriesPlot.set_xlabel('Time (seconds)')
    timeSeriesPlot.set_ylabel('Amplitude (dB)')
    # timeSeriesPlot.legend()

    fftSize = 1024
    spectrogramPlot.specgram(scan.getMag(), Fs=settings['sampleRate'], NFFT=fftSize, noverlap=int(fftSize/2))
    spectrogramPlot.set_xlabel('Time (seconds)')
    spectrogramPlot.set_ylabel('Frequency (Hz)')
    if show:
        plt.show()
        gc.collect() # plots don't release memory on their own for some reason
    

def amplitudePlot(scan: Scan, settings, show=True):
    data = util.lin2db(scan.getMag())
    data = np.clip(data, np.median(data), np.inf) # Clip bottom to the noise floor
    plt.plot(scan.getDataTimes() - scan.tStart, data)
    for tDet in scan.targetTimes:
        plt.axvline(tDet - scan.tStart, color='r', linestyle='--')

    plt.title('Time vs Amplitude')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (dB)')
    if show:
        plt.show()
        gc.collect() # plots don't release memory on their own for some reason

mapImage = np.array(Image.open('map.jpg'))
def plotReturns(returns: List[List[RadReturn]], targets: List[np.ndarray], settings, show=True):
    plt.imshow(mapImage)
    
    transmitterXY = LL2pixel(*settings['transmitterLLA'][0:2])
    receiverXY = LL2pixel(*settings['receiverLLA'][0:2])
    if transmitterXY is None or receiverXY is None:
        return

    targetXY = np.array([LL2pixel(LLA[0], LLA[1]) for LLA in targets])

    plt.scatter(transmitterXY[0], transmitterXY[1], label='Transmitter')
    plt.scatter(receiverXY[0], receiverXY[1], label='Receiver')
    if targetXY.shape != (0,):
        plt.scatter(targetXY[:,0], targetXY[:,1], label='Targets')
    for target in targets:
        XY = LL2pixel(target[0], target[1])
        if XY is None:
            continue
        plt.text(XY[0], XY[1], f"{target[2]/1e3:.2f}")

    for scanIdx, scanReturns in enumerate(returns):
        retXY = [LL2pixel(r.LLA[0], r.LLA[1]) for r in scanReturns]
        retXY = np.array(list(filter(lambda x: x is not None, retXY)))
        plt.scatter(retXY[:,0], retXY[:,1], label=f'Returns {scanIdx}', marker='.')
        for r in scanReturns:
            XY = LL2pixel(r.LLA[0], r.LLA[1])
            if XY is None:
                continue
            plt.text(XY[0], XY[1], f"{r.LLA[2]/1e3:.2f}")

    if show:
        plt.legend()
        plt.show()
        gc.collect() # plots don't release memory on their own for some reason
    

LAT_DIM = 0
LON_DIM = 1
mapGeodeticDims = [[34.25, 33.875], [-117.8750, -117.3750]]
mapPixelDims = [[0, 9000], [0, 10000]]

def LL2pixel(lat, lon) -> np.ndarray | None:
    percY = util.inv_lerp(mapGeodeticDims[LAT_DIM][0], mapGeodeticDims[LAT_DIM][1], lat)
    percX = util.inv_lerp(mapGeodeticDims[LON_DIM][0], mapGeodeticDims[LON_DIM][1], lon)

    if percX < 0 or percX > 1:
        return None
    if percY < 0 or percY > 1:
        return None

    y = util.lerp(mapPixelDims[LAT_DIM][0], mapPixelDims[LAT_DIM][1], percY)
    x = util.lerp(mapPixelDims[LON_DIM][0], mapPixelDims[LON_DIM][1], percX)

    return np.array([x, y])

def pixel2LL(x, y) -> np.ndarray | None:
    percY = util.inv_lerp(mapPixelDims[LAT_DIM][0], mapPixelDims[LAT_DIM][1], y)
    percX = util.inv_lerp(mapPixelDims[LON_DIM][0], mapPixelDims[LON_DIM][1], x)

    if percX < 0 or percX > 1:
        return None
    if percY < 0 or percY > 1:
        return None

    lat = util.lerp(mapGeodeticDims[LAT_DIM][0], mapGeodeticDims[LAT_DIM][1], percY)
    lon = util.lerp(mapGeodeticDims[LON_DIM][0], mapGeodeticDims[LON_DIM][1], percX)

    return np.array([lat, lon])

