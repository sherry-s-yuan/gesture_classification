import numpy as np
import pickle
from joblib import dump, load
# normalization scaler storage
x_scaler = load('x_scaler.joblib')
zc_scaler = load('zc_scaler.joblib')
ssc_scaler = load('ssc_scaler.joblib')
oned_scaler = load('oned_scaler.joblib')

# with open("x_scaler.pickle") as f:
#     x_scaler = pickle.load(f)
# with open("zc_scaler.pickle") as f:
#     zc_scaler = pickle.load(f)
# with open("ssc_scaler.pickle") as f:
#     ssc_scaler = pickle.load(f)
# with open("oned_scaler.pickle") as f:
#     oned_scaler = pickle.load(f)

""" Labels
0 - unmarked data,
1 - hand at rest, 
2 - hand clenched in a fist, 
3 - wrist flexion,
4 – wrist extension,
5 – radial deviations,
6 - ulnar deviations,
7 - extended palm (the gesture was not performed by all subjects).
"""

"""
Distinguishing factor (0 = low, 1 = high, 2 = very high)
| channel\class | class 1 | class 2 | class 3 | class 4 | class 5 | class 6 | class 7 |
|---------------|---------|---------|---------|---------|---------|---------|---------|
| channel1      | 0       | 2       | 1       | 0       | 0       | 2       |         |
| channel2      | 0       | 1       | 1       | 0       | 1       | 1       |         |
| channel3      | 0       | 1       | 0       | 0       | 1       | 0       |         |
| channel4      | 0       | 1       | 0       | 1       | 1       | 0       |         |
| channel5      | 0       | 1       | 1       | 2       | 2       | 1       |         |
| channel6      | 0       | 1       | 0       | 1       | 1       | 0       |         |
| channel7      | 0       | 1       | 0       | 0       | 0       | 1       |         |
| channel8      | 0       | 1       | 1       | 0       | 0       | 1       |         |
"""


class Signal:
    def __init__(self, data):
        self.x = np.array(data)
        self.ssc_feature = np.array(self.SSC(self.x))
        self.zc_feature = np.array(self.ZC(self.x))
        self.feature_1d = np.array(self.feature_extraction_1D(self.x))

    def MAV(self, x):
        """compute mean absolute value, return (1, channel)"""
        return np.mean(np.abs(x), axis=0) #pd.DataFrame(x.abs().mean(axis=0)).T

    def ZC(self, x):
        """compute zero crossing, return (window, channel)"""
        a2 = np.sign(x)
        change2 = ((np.roll(a2, 1, axis=0) - a2) != 0).astype(int)
        return change2 #change2.to_numpy()

    def SSC(self, x):
        """detect slope change, return (window-1, channel)"""
        a = np.sign(np.diff(x, axis=0))
        change = ((np.roll(a, 1, axis=0) - a) != 0).astype(int)
        return change

    def WL(self, x):
        """waveform length feature extraction, return (1, channel)"""
        # columns = x.columns
        a = np.sum(np.diff(x, axis=0), axis=0)
        # change = pd.DataFrame(a).T
        # change.columns = columns
        return a #change

    def STD(self, x):
        """standard deviation of the channels, return (1, channel)"""
        return x.std(axis=0) # pd.DataFrame(x.std(axis=0)).T

    def RMS(self, x):
        """Root mean squared, return (1, channel)"""
        return ((x ** 2).sum(axis=0) / x.shape[0]) ** (1 / 2) # pd.DataFrame(((x ** 2).sum(axis=0) / x.shape[0]) ** (1 / 2)).T

    def feature_extraction_1D(self, x):
        """join all 1D feature together"""
        rms = [self.RMS(x)]
        wl = self.WL(x)
        std = self.STD(x)
        mav = self.MAV(x)
        rms.append(wl)
        rms.append(std)
        rms.append(mav)
        return np.array(rms)


class Data:
    def __init__(self, data):
        """the function is expecting input of shape (1800, 8)
        where 1800 is the time, and 8 is the channel"""
        data = Signal(data[0,:,:])
        self.X = np.array([data.x])
        self.zc_feature = np.array([data.zc_feature])
        self.ssc_feature = np.array([data.ssc_feature])
        self.feature_1d = np.array([data.feature_1d])
        self.normalize()

    def normalize(self):
        for i in range(self.X.shape[1]):
            self.X[:, i, :] = x_scaler[i].transform(self.X[:, i, :])
        for i in range(self.zc_feature.shape[1]):
            self.zc_feature[:, i, :] = zc_scaler[i].transform(self.zc_feature[:, i, :])
        for i in range(self.ssc_feature.shape[1]):
            self.ssc_feature[:, i, :] = ssc_scaler[i].transform(self.ssc_feature[:, i, :])
        for i in range(self.feature_1d.shape[1]):
            self.feature_1d[:, i, :] = oned_scaler[i].transform(self.feature_1d[:, i, :])



