import numpy as np

ASR11_SCAN_RATE = 60/13 # Seconds per revolution
ASR11_PULSE_RATE = 1e-3 # Seconds
ASR11_PULSE_WIDTH = 1e-6 # Seconds

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


