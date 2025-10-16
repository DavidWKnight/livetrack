import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import pymap3d
import scipy

import util

def maxPool1d(data: np.ndarray, stride: int) -> np.ndarray:
    """Downsample by taking the max of neighbors"""
    return np.array([max(chunk) for chunk in util.chunks(data, stride)])

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
    expectedPulsesPerWindow = util.ASR11_SCAN_RATE / (1/newSampleRate)
    percent = (1 - (1/expectedPulsesPerWindow)) * 100
    margin = 0.02 # percent, determined experimentally
    percent = percent * (1-margin)
    threshold = np.percentile(down, percent)

    # Threshold the input and find centers
    mask = down > threshold
    clusters, numClusters = ndimage.label(mask)
    centers = np.array([c[0] for c in ndimage.center_of_mass(down, clusters, range(1,numClusters+1))])
    centers = centers * (decimate/sampleRate)
    
    # Clean up center locations based on knowledge of how the ASR11 system works
    centers = refineCenters(data, centers, sampleRate, decimateTime, resolution, plot)
    centers = filterPulses(centers, len(data)/sampleRate)
    centers = refineCenters(data, centers, sampleRate, decimateTime, resolution, plot)
    centerIndexes = (centers * sampleRate).astype(np.int64)

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

def refineCenters(data, centers, sampleRate, decimateTime, resolution, plot):
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
    return np.array(refinedCenters)

def filterPulses(centers: np.ndarray, tMax: float):
    # Try to skip over detected pulses to see if it would result in a pulse closer to the expected time
    # Do this by finding the indexes that most likely align to the ASR11 frequency and then removing those that don't line up
    diff = np.diff(centers)
    rightPlace = [util.isCloseMultiple(d, util.ASR11_SCAN_RATE, 0.2) for d in diff]
    rightPlace.append(True)
    centers = centers[rightPlace]


    # Extend the centers array out to the beginning of the time range
    nPulsesToInsert = np.floor(centers[0] / util.ASR11_SCAN_RATE)
    pulseOffsets = np.array(range(1, int(nPulsesToInsert)+1)) * util.ASR11_SCAN_RATE
    newCenters = centers[0] - pulseOffsets[::-1]
    centers = np.hstack([newCenters, centers])

    # Extend the centers array out to the end of the time range
    nPulsesToInsert = np.floor((tMax-centers[-1]) / util.ASR11_SCAN_RATE)
    b = range(1, int(nPulsesToInsert)+1)
    a = np.array(b)
    pulseOffsets = a * util.ASR11_SCAN_RATE
    newCenters = centers[-1] + pulseOffsets
    centers = np.hstack([centers, newCenters])

    # If we skipped a pulse we should see another some integer multiple of the scan rate later
    # If we detect this try splitting it up
    newCenters = [centers[0]]
    previous = centers[0]
    for c in centers[1:]:
        diff = c - previous
        rightPlace = util.isCloseMultiple(diff, util.ASR11_SCAN_RATE, 0.25)
        bigEnough = diff > util.ASR11_SCAN_RATE*1.5

        if rightPlace and bigEnough:
            nSkippedPulses = np.round(diff / util.ASR11_SCAN_RATE)
            missing = np.linspace(previous, c, int(nSkippedPulses)+1)
            missing = missing[1:] # Skip the 'previous' point, it's already been added
            newCenters.extend(missing)
        else:
            newCenters.append(c)
        previous = c
    centers = np.array(newCenters)

    return np.array(centers)

def findPulseTime(data: np.ndarray, sampleRate: float, resolution: int=1, plot: bool=False):# 
    """
    Find the center time of several ASR11 pulses
    """
    # This could probably be smarter by doing a weighted center of mass based on each pulse
    # But this should be good enough for now
    # Center does not need to align with a pulse since it's actually unlikely that the antenna was pointed directly at the pulse time

    threshold = np.percentile(data, 99.98)
    mask = data > threshold
    mask = ndimage.binary_dilation(mask, [True, True, True, True, True])
    clusters, numClusters = ndimage.label(mask)
    centers = np.array([c[0] for c in ndimage.center_of_mass(data, clusters, range(1,numClusters+1))])
    intensities = data[np.uint64(np.round(centers))]
    weightedCenter = int(round(np.sum(intensities*centers) / np.sum(intensities)))

    idx = np.argmax(data)

    if False:
        plt.plot(data)
        plt.scatter(centers, intensities)
        plt.axvline(weightedCenter, color='red')
        plt.axvline(idx, color='green')
        plt.axhline(threshold)
        plt.show()
    return weightedCenter / sampleRate, weightedCenter

def matchFilter(data: np.ndarray, sampleRate: float) -> np.ndarray:
    matchLenth = int(np.ceil(sampleRate * util.ASR11_PULSE_WIDTH))
    kernel = np.ones(matchLenth)
    return np.convolve(data, kernel, 'valid')

def pulseIntegration(data: np.ndarray, sampleRate: float, numIntegrations: int=10) -> np.ndarray:
    fastFrameSampleSize = int(sampleRate/1000)
    fastFrames = util.chunks(data, fastFrameSampleSize)
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

# def movingTargetIndicator(data: np.ndarray, sampleRate: float, scanStartSample):
#     slowFrameSize = int(util.ASR11_SCAN_RATE * sampleRate)
#     slowFrames = list(chunks(data[scanStartSample:], slowFrameSize))
    
#     output = data[scanStartSample:] # Retain the stuff from before the scan started
#     for previousFrame, frame in zip(slowFrames[:-1], slowFrames[1:]):
#         if len(frame) != slowFrameSize:
#             output.extend(frame) # Likely the last samples in the collect
#             continue

#         if len(output) == 0:
#             output.extend(previousFrame) # Retain the first slow frame that can't have MTI on it
#         output.extend(np.array(frame) - np.array(previousFrame))
#     return output

def cfar(X_k, num_guard_cells, num_ref_cells, bias, cfar_method="average"):
    N = X_k.size
    cfar_values = np.zeros(X_k.shape)
    for center_index in range(
        num_guard_cells + num_ref_cells, N - (num_guard_cells + num_ref_cells)
    ):
        min_index = center_index - (num_guard_cells + num_ref_cells)
        min_guard = center_index - num_guard_cells
        max_index = center_index + (num_guard_cells + num_ref_cells) + 1
        max_guard = center_index + num_guard_cells + 1

        lower_nearby = X_k[min_index:min_guard]
        upper_nearby = X_k[max_guard:max_index]

        lower_mean = np.mean(lower_nearby)
        upper_mean = np.mean(upper_nearby)

        if cfar_method == "average":
            mean = np.mean(np.concatenate((lower_nearby, upper_nearby)))
        elif cfar_method == "greatest":
            mean = max(lower_mean, upper_mean)
        elif cfar_method == "smallest":
            mean = min(lower_mean, upper_mean)
        else:
            mean = 0

        output = mean * bias
        cfar_values[center_index] = output

    targets_only = np.copy(X_k)
    targets_only[np.where(X_k < cfar_values)] = np.ma.masked

    return cfar_values, targets_only

def bistaticRange2ElRange(transmitterLLA, receiverLLA, bistaticRange, az):
    transmitterECEF = np.array(pymap3d.geodetic2ecef(*transmitterLLA))
    receiverECEF = np.array(pymap3d.geodetic2ecef(*receiverLLA))

    def distanceError(a):
        (r, el) = a
        estTargetECEF = np.array(pymap3d.aer2ecef(az, el, r, *transmitterLLA))
        d1 = np.linalg.norm(estTargetECEF - transmitterECEF)
        d2 = np.linalg.norm(estTargetECEF - receiverECEF)
        return abs(bistaticRange - (d1 + d2))
    
    distMin = 0
    distMax = distMin + (bistaticRange - distMin)/2
    bnds = ((distMin, distMax), (1, 5))
    x0 = ((distMin+distMax)/2, 2)
    res = scipy.optimize.minimize(distanceError, x0, method='TNC', bounds=bnds, tol=1e-10)
    return res.x
