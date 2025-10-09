
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


while not RFDataManager.isEndOfFile():
    scan = RFDataManager.getNextScan()
    scan.appendTarget(aircraftState)
    amplitudeSpectrogramPlot(scan, settings)

# previousScan = RFDataManager.getNextScan()
# previousScan.appendTarget(aircraftState)
# while not RFDataManager.isEndOfFile():
#     scan = RFDataManager.getNextScan()
#     scan.appendTarget(aircraftState)
#     amplitudePlot(previousScan, settings, False)
#     amplitudePlot(scan, settings)

