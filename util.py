import numpy as np

ASR9_SCAN_RATE = 60/13 # Seconds per revolution
ASR9_LONG_PULSE_INTERVAL = 0.99803435e-3 # Seconds
ASR9_SHORT_PULSE_INTERVAL = 0.77980219e-3 # Seconds
ASR9_PULSE_WIDTH = 1e-6 # Seconds

ASR9_AZ_STD = (1.4*1.5/3)
ASR9_EL_STD = (3/3)
ASR9_RANGE_STD = (500 / 3)

# Code to find the pulse interval
# [frameTimes, _] = scan.getFramesTimes()
# betterFrameTimes = [frameTimes[0]]
# for f in frameTimes[1:]:
#     if f - betterFrameTimes[-1] < 500e-6:
#         continue
#     betterFrameTimes.append(f)

# bins = np.linspace(0, 5e-3, 10000)
# [data, dataEdges] = np.histogram(np.diff(betterFrameTimes), bins)
# print(data)
# mask = data > 5
# mask = ndimage.binary_dilation(mask, [True, True, True, True, True])
# clusters, numClusters = ndimage.label(mask)
# centers = np.array([c[0] for c in ndimage.center_of_mass(data, clusters, range(1,numClusters+1))])
# print(centers * (5e-3 / 10000) * 1000)
# plt.plot(dataEdges[1:], data)
# plt.axhline(5)
# plt.show()

def nearestIdx(arr, b):
    """Find which index into arr is closest to the value b."""
    return np.abs(arr - b).argmin()

def padSum(a, b):
    """Sums up a and by similar to np.sum but retains the shape of a"""
    if len(a) == len(b):
        return a + b
    elif len(a) > len(b):
        bPad = np.pad(b, (0, len(a) - len(b)))
        return a + bPad
    else:
        aPad = np.pad(a, (0, len(b) - len(a)))
        return (aPad + b)[:len(a)]

def padSub(a, b):
    """Subtracks b from a np.subtract but retains the shape of a"""
    if len(a) == len(b):
        return a - b
    elif len(a) > len(b):
        bPad = np.pad(b, (0, len(a) - len(b)))
        return a - bPad
    else:
        aPad = np.pad(a, (0, len(b) - len(a)))
        return (aPad - b)[:len(a)]

def lin2db(x):
    return 20 * np.log10(x + np.finfo(float).eps)

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    Examples
    --------
        50 == lerp(0, 100, 0.5)
        4.2 == lerp(1, 5, 0.8)
    """
    # https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
    return (1 - t) * a + t * b

def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    # https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
    return (v - a) / (b - a)

def isCloseMultiple(a, b, atol):
    multiples = a / b
    diff = multiples - round(multiples)
    return abs(diff*b) < atol

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    # https://stackoverflow.com/a/312464
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


