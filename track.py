import numpy as np

class Data(object):

    def __init__(self, dataPath):
        self.data = np.loadtxt(dataPath, delimiter=',', skiprows=1)

        # Turn gyroscope deg/s to rad/s
        self.data[:, (1,2,3)] *= np.pi / 180
    
    def getRot(self, i):
        return self.data[i, (1, 2, 3)]
    
    def getAcc(self, i):
        return self.data[i, (4, 5, 6)]
    
    def getMag(self, i):
        return self.data[i, (7, 8, 9)]