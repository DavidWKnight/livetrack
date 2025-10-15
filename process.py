
import numpy as np
import matplotlib.pyplot as plt

import dataLoad

from plotData import amplitudeSpectrogramPlot
from plotData import amplitudePlot

def lin2db(x):
    return 20 * np.log10(x + np.finfo(float).eps)

folder = '/media/david/Ext/Collects/**/'
# fnameBase = '2025-10-04T12:08:50_ABC504'
# fnameBase = '2025-09-28T17:06:03_A3AA32'
fnameBase = '2025-10-06T17:26:46_AC98AC'
# fnameBase = '2025-09-28T17:17:35_A420AF'

aircraftState, settings, RFDataManager = dataLoad.loadCollect(folder + fnameBase)

# previous = RFDataManager.getNextScan()
# while not RFDataManager.isEndOfFile():
#     scan = RFDataManager.getNextScan()
#     scan.appendTarget(aircraftState)
    
#     pulseLines, threshold = scan.getFramesTimes()
#     plt.plot(np.diff(pulseLines))
#     plt.show()

#     plt.hist(np.diff(pulseLines), np.arange(0.6e-3, 1.1e-3, 1e-6))
#     plt.show()

#     # scan.getClutterSuppressedMag()

#     amplitudePlot(scan, settings, False)
#     # amplitudePlot(scan.applyPulseIntegration(), settings, False)
#     # amplitudePlot(previous, settings, False)
#     plt.show()

#     previous = scan

while not RFDataManager.isEndOfFile():
    scan = RFDataManager.getNextScan()
    scan.appendTarget(aircraftState)
    frames = scan.toFrames()
    print(len(frames))
    returns = frames[1000].getReturns()
    for r in returns:
        print(r)
    input('Hello')
    amplitudeSpectrogramPlot(scan.applyPulseIntegration(), settings)

