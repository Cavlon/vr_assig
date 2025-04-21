import numpy as np

class Data(object):

    def __init__(self, dataPath):
        """ Holds all input data from a csv file
            Converts rotational deg/s to rad/s
        """
        self.data = np.loadtxt(dataPath, delimiter=',', skiprows=1)

        # Turn gyroscope deg/s to rad/s
        self.data[:, (1,2,3)] *= np.pi / 180

        # Z is up in the data but Y is up in the program so fields need to be swapped
        self.data[:, (1, 2, 3)] = self.data[:, (2, 3, 1)]
        self.data[:, (4, 5, 6)] = self.data[:, (5, 6, 4)]
        self.data[:, (7, 8, 9)] = self.data[:, (8, 9, 7)]

    def getTime(self, i):
         return self.data[i, 0]

    def getRot(self, i):
        return self.data[i, (1, 2, 3)]
    
    def getAcc(self, i):
        return self.data[i, (4, 5, 6)]
    
    def getMag(self, i):
        return self.data[i, (7, 8, 9)]