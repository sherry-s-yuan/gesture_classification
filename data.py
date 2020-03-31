import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#
# pd.set_option("display.precision", 8)

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
    def __init__(self, path):
        self.data = pd.read_csv(path, delimiter='\t', encoding='UTF-8')
        # filter out class 0 and 7
        self.data = self.data.loc[self.data['class'] != 7]
        self.data = self.data.loc[self.data['class'] != 0]
        # load in data
        self.label = self.data['class'].to_numpy()
        self.time = self.data['time']
        self.data = self.data.iloc[:, 1:-1]
        self.filename = path.split('/')[-1].split('.txt')[0]

    def generate(self, step=100, window=1800):
        """break time series data into periods of windows of 0.5 seconds for detection"""
        self.x = []
        self.y = []
        for i in range(0, self.data.shape[0], step):
            if self.data.shape[0] - i < 1800:
                break
            # Generate one hot encoding of labels
            label = [0,0,0,0,0,0,0]
            majority_class = self._majority_class(self.label[i:i+window])
            if majority_class == 0:
                continue
            label[majority_class] = 1
            self.y.append(label)
            # prepare x
            self.x.append(self.data[i:i+window, :])
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def feature_generation(self):
        """prepare all ssc, zc, average, rms etc features"""
        self.ssc_feature = []
        self.zc_feature = []
        self.feature_1d = []
        for timeframe in self.x:
            self.ssc_feature.append(self.SSC(timeframe))
            self.zc_feature.append(self.ZC(timeframe))
            self.feature_1d.append(self.feature_extraction_1D(timeframe))
        self.ssc_feature = np.array(self.ssc_feature)
        self.zc_feature = np.array(self.zc_feature)
        self.feature_1d = np.array(self.feature_1d)

    def plot(self):
        """plot raw signal with class labels as colored region"""
        fig, ax = plt.subplots()
        max_num = self.data.max().max()
        min_num = self.data.min().min()
        color = ['', 'b', 'g', 'r', 'c', 'm', 'y', 'violet']
        # color above correspond to class, blue correspond to class 1 etc.
        for label in range(1,7):
            ax.fill_between(self.time, min_num, max_num, where=self.label == label,
                        facecolor=color[label], alpha=0.3)
        for i in range(1, 9):
            ax.plot(self.time, self.data['channel' + str(i)])
        plt.show()

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

    def _majority_class(self, labels):
        """if a time window contains more than 60% of a class, then that window will be labelled as that class"""
        label_count = []
        for label in range(0, 8):
            label_count.append(np.count_nonzero(labels == label))# labels.count(label))
        if max(label_count)/sum(label_count) < 0.65:
            return 0
        else:
            return label_count.index(max(label_count))

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

    def to_numpy(self):
        self.data = self.data.to_numpy()

    def save_data(self):
        np.save("./data/" + self.filename + '_y', self.y)
        np.save("./data/" + self.filename + '_x', self.x)
        np.save("./data/" + self.filename + '_zc', self.zc_feature)
        np.save("./data/" + self.filename + '_ssc', self.ssc_feature)
        np.save("./data/" + self.filename + '_1d', self.feature_1d)

    def load_data(self):
        y = np.load("./data/" + self.filename + '_y' + ".npy")
        x = np.load("./data/" + self.filename + '_x' + ".npy")
        zc_feature = np.load("./data/" + self.filename + '_zc' + ".npy")
        ssc_feature = np.load("./data/" + self.filename + '_ssc' + ".npy")
        feature_1d = np.load("./data/" + self.filename + '_1d' + ".npy")
        return x, zc_feature, ssc_feature, feature_1d, y

# # To go through the whole data preparation process, run following
# signal = Signal('asdf')
# # signal.plot()
# signal.to_numpy()
# signal.generate()
# signal.feature_generation()
# signal.save_data()
# signal.load_data()

## Testing purpose, ignore
# print(signal.MAV(signal.x[0]))
# print(signal.ZC(signal.x[0]).shape)
# print(signal.SSC(signal.x[0]).shape)
# print(signal.WL(signal.x[0]))
# print(signal.STD(signal.x[0]))
# print(signal.RMS(signal.x[0]))
# print(signal.feature_extraction_1D(signal.x[0]).shape)
# print(signal.x.shape)
# print(signal.zc_feature.shape)
# print(signal.ssc_feature.shape)
# print(signal.feature_1d.shape)
# print(signal.y.shape)
# print(signal.data)

class Data:
    def __init__(self, X, zc_features, ssc_features, features_1d, Y):
        print("Train Test Split:")
        # get training dataset index
        train_ind = np.random.choice(np.arange(X.shape[0]), int(X.shape[0] * 0.7), replace=False)
        test_ind = np.setdiff1d(np.arange(X.shape[0]), train_ind)
        # training data
        self.X_train = np.take(X, train_ind, axis=0)
        self.zc_features_train = np.take(zc_features, train_ind, axis=0)
        self.ssc_features_train = np.take(ssc_features, train_ind, axis=0)
        self.features_1d_train = np.take(features_1d, train_ind, axis=0)
        self.Y_train = np.take(Y, train_ind, axis=0)
        # test data
        self.X_test = np.take(X, test_ind, axis=0)
        self.zc_features_test = np.take(zc_features, test_ind, axis=0)
        self.ssc_features_test = np.take(ssc_features, test_ind, axis=0)
        self.features_1d_test = np.take(features_1d, test_ind, axis=0)
        self.Y_test = np.take(Y, test_ind, axis=0)
        # normalization scaler storage
        self.x_scaler = {}
        self.zc_scaler = {}
        self.ssc_scaler = {}
        self.oned_scaler = {}
        self.normalize_train_test() # normalize data using StandardScaler

    def normalize_train_test(self):
        # using scaler fitted on training data on test data
        if len(self.x_scaler) == 0:
            # fit transform training set
            for i in range(self.X_train.shape[1]):
                self.x_scaler[i] = StandardScaler()
                self.X_train[:, i, :] = self.x_scaler[i].fit_transform(self.X_train[:, i, :])
            for i in range(self.zc_features_train.shape[1]):
                self.zc_scaler[i] = StandardScaler()
                self.zc_features_train[:, i, :] = self.zc_scaler[i].fit_transform(self.zc_features_train[:, i, :])
            for i in range(self.ssc_features_train.shape[1]):
                self.ssc_scaler[i] = StandardScaler()
                self.ssc_features_train[:, i, :] = self.ssc_scaler[i].fit_transform(self.ssc_features_train[:, i, :])
            for i in range(self.features_1d_train.shape[1]):
                self.oned_scaler[i] = StandardScaler()
                self.features_1d_train[:, i, :] = self.oned_scaler[i].fit_transform(self.features_1d_train[:, i, :])
        # fit transform test
        for i in range(self.X_test.shape[1]):
            self.X_test[:, i, :] = self.x_scaler[i].transform(self.X_test[:, i, :])
        for i in range(self.zc_features_test.shape[1]):
            self.zc_features_test[:, i, :] = self.zc_scaler[i].transform(self.zc_features_test[:, i, :])
        for i in range(self.ssc_features_test.shape[1]):
            self.ssc_features_test[:, i, :] = self.ssc_scaler[i].transform(self.ssc_features_test[:, i, :])
        for i in range(self.features_1d_test.shape[1]):
            self.features_1d_test[:, i, :] = self.oned_scaler[i].transform(self.features_1d_test[:, i, :])

    def normalize(self, X, zc_features, ssc_features, features_1d):
        """this function manipute the variables directly: aliasing"""
        for i in range(self.X_test.shape[1]):
            X[:, i, :] = self.x_scaler[i].transform(X[:, i, :])
        for i in range(self.zc_features_test.shape[1]):
            zc_features[:, i, :] = self.zc_scaler[i].transform(zc_features[:, i, :])
        for i in range(self.ssc_features_test.shape[1]):
            ssc_features[:, i, :] = self.ssc_scaler[i].transform(ssc_features[:, i, :])
        for i in range(self.features_1d_test.shape[1]):
            features_1d[:, i, :] = self.oned_scaler[i].transform(features_1d[:, i, :])
        return X, zc_features, ssc_features, features_1d



