import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

ASR11_SCAN_RATE = 4.6 # Seconds per revolution
ASR11_PULSE_RATE = 1e-3 # Seconds
ASR11_PULSE_WIDTH = 1e-6 # Seconds

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def maxPool1d(data: np.ndarray, stride: int) -> np.ndarray:
    """Downsample by taking the max of neighbors"""
    return np.array([max(chunk) for chunk in chunks(data, stride)])

def findDirectPathPulses(data: np.ndarray, sampleRate: float, resolution=1, plot: bool=False):
    """
    Find the pulse centers
    
    :param data: RF amplitude data
    :type data: np.array
    :param sampleRate: Sample rate of the SDR
    :type sampleRate: int
    :param resolution: Angular resolution to resolve
    :type resolution: float

    :return: An array of center times and the array indexes into data
    :rtype: tuple - (np.array, np.array)
    """
    # We will start by finding the bin centers at a very low angular resolution

    # Downsample the input signal to make it easier to find peaks
    decimateTime = 0.2
    decimate = decimateTime * sampleRate # Only about (0.2/4.8)*360 = 15 degrees resolution
    down = maxPool1d(data, round(decimate))
    newSampleRate = sampleRate / decimate
    
    # Calculate the threshold to use to find peaks
    expectedPulsesPerWindow = ASR11_SCAN_RATE / (1/newSampleRate)
    percent = (1 - (1/expectedPulsesPerWindow)) * 100
    margin = 0.02 # percent, determined experimentally
    percent = percent * (1-margin)
    threshold = np.percentile(down, percent)

    # Threshold the input and find centers
    mask = down > threshold
    clusters, numClusters = ndimage.label(mask)
    centers = np.array([c[0] for c in ndimage.center_of_mass(down, clusters, range(1,numClusters+1))])
    centers = centers * (decimate/sampleRate)
    
    refinedCenters = []
    centerIndexes = []
    for c in centers:
        windowStart = c - decimateTime
        windowEnd = c + decimateTime
        windowStartIdx = np.int64(windowStart * sampleRate)
        windowEndIdx = np.int64(windowEnd * sampleRate)
        window = data[windowStartIdx:windowEndIdx]
        newCenter, newCenterIdx = findPulseTime(window, sampleRate, resolution, plot)
        refinedCenters.append(newCenter + windowStart)
        centerIndexes.append(newCenterIdx + windowStartIdx)
    centers = np.array(refinedCenters)

    # plot for debugging
    if plot:
        print(centers)
        print(np.diff(centers))

        decimateTimeResolution = (decimate/sampleRate)
        binTimes = np.array(range(0, len(down))) * decimateTimeResolution
        downCenterIndexes = np.int64(np.round(centers / (decimate/sampleRate)))
        centerAmplitudes = down[downCenterIndexes]
        plt.plot(binTimes, down)
        plt.axhline(y=threshold, color='red', linestyle='--')
        plt.scatter(centers, centerAmplitudes)
        plt.show()
        # Note that the scatter plot may not align with the line plot since the line plot is downsampled

    return np.array(centers), np.array(centerIndexes)

def findPulseTime(data: np.ndarray, sampleRate: float, resolution: int=1, plot: bool=False):# 
    """
    Find the center time of several ASR11 pulses
    """
    # This could probably be smarter by doing a weighted center of mass based on each pulse
    # But this should be good enough for now

    idx = np.argmax(data)

    if plot:
        plt.plot(data)
        plt.scatter(idx, data[idx])
        plt.show()
    return idx / sampleRate, idx

def matchFilter(data: np.ndarray, sampleRate: float) -> np.ndarray:
    matchLenth = int(np.ceil(sampleRate * ASR11_PULSE_WIDTH))
    kernel = np.ones(matchLenth)
    return np.convolve(data, kernel, 'valid')

def pulseIntegration(data: np.ndarray, sampleRate: float, numIntegrations: int=10) -> np.ndarray:
    fastFrameSampleSize = int(sampleRate/1000)
    fastFrames = chunks(data, fastFrameSampleSize)
    runningSum = [[0]*fastFrameSampleSize]*numIntegrations

    circBufferIdx = 0
    output = []
    for frame in fastFrames:
        if len(frame) != fastFrameSampleSize:
            output.extend(frame) # Likely the last samples in the collect
            continue

        runningSum[circBufferIdx] = frame
        output.extend(np.sum(runningSum, 0))
        circBufferIdx = (circBufferIdx + 1) % numIntegrations

    return output

