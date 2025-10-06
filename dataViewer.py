import gzip

import numpy as np
import matplotlib.pyplot as plt

sampleRate = 3e6

def getNextChunk(fname, chunkSize=4.6*sampleRate):
    bytesPerSample = 4
    readSize = (bytesPerSample * chunkSize)
    readSize = int(readSize - (readSize % bytesPerSample))
    
    with gzip.open(fname, 'rb') as file:
        while True:
            a = file.read(readSize)
            if a == b'':
                break
            
            data = np.frombuffer(a, np.int16)
            data = data[::2] + 1j * data[1::2]
            yield data

# folder = '/media/david/Ext/Collects/collect_10_4/collects2/'
# fname = '2025-10-04T14:57:07_AA7B54.dat.gz'
folder = ''
fname = 'collect.dat.gz'
for chunk in getNextChunk(folder + fname):
    data = np.absolute(chunk)
    plt.plot(np.array(range(len(data))) / sampleRate, data)
    plt.show()
