
import numpy as np
import matplotlib.pyplot as plt

import dataLoad
from rfTypes import collect2scans

def lin2db(x):
    return 20 * np.log10(x)

folder = '/media/david/Ext/Collects/**/'
# fnameBase = '2025-10-04T12:08:50_ABC504'
# fnameBase = '2025-09-28T17:06:03_A3AA32'
fnameBase = '2025-10-06T17:26:46_AC98AC'
# fnameBase = '2025-09-28T17:17:35_A420AF'

aircraftState, settings, data = dataLoad.loadCollect(folder + fnameBase)
data = np.absolute(data)

scans = collect2scans(data, settings)

for scan in scans:
    scan.appendTarget(aircraftState)

for scan in scans:
    plt.plot(scan.getDataTimes() - scan.tStart, lin2db(scan.data))
    for tDet in scan.targetTimes:
        plt.axvline(tDet - scan.tStart,  color='r', linestyle='--')
    plt.show()
