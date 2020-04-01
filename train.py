import os
from data import Signal, Data
import numpy as np
from model import BILSTM
import pickle

# Prepare paths to data access
print("Prepare Paths:")
paths = []
repository = "./raw/"
directories = []
files = []
for r, d, f in os.walk(repository):
    directories += d
    if len(f) != 0:
        files.append(f)

for i in range(len(directories)):
    directory = directories[i]
    for file in files[i]:
        paths.append(repository + directory + '/' + file)

# Preprocessing Signal File
print("Processing Signal Files:")
signals = []
count = 0
for path in paths:
    # if count >= 14:
    #     break
    print(path)
    signal = Signal(path)
    signal.to_numpy()
    signal.generate()
    signal.feature_generation()
    signal.save_data()
    signals.append(signal)
    count += 1

# Concatenate Train Test Data
print("Concatenate Train, Test Data:")
X, zc_features, ssc_features, features_1d, Y = signals[0].load_data()
for signal in signals[1:]:
    x, zc_feature, ssc_feature, feature_1d, y = signal.load_data()
    X = np.append(X, x, axis=0)
    zc_features = np.append(zc_features, zc_feature, axis=0)
    ssc_features = np.append(ssc_features, ssc_feature, axis=0)
    features_1d = np.append(features_1d, feature_1d, axis=0)
    Y = np.append(Y, y, axis=0)


def get_class(predictions):
    """return class label base on one hot encoding"""
    result = []
    for prediction in predictions:
        prediction = prediction.tolist()
        result.append(prediction.index(max(prediction)))
    return np.array(result)

# from sklearn.multiclass import OutputCodeClassifier
# from sklearn.svm import LinearSVC
# clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
# fake_x = features_1d.reshape((features_1d.shape[0], features_1d.shape[1]*features_1d.shape[2]))
# clf.fit(fake_x[1:,:], get_class(Y[1:,:]))
# print(clf.predict(np.array([fake_x[0,:]])))
# print(Y)
# print(get_class(Y[0,:]))


# Prepare Data
data = Data(X, zc_features, ssc_features, features_1d, Y)

# dump data
with open('data.pickle', 'wb') as f:
    pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)

# # reload data (uncomment this only if you have data.pickle in the directory)
# with open('data.pickle', 'rb') as f:
#     data = pickle.load(f)

model = BILSTM()

# do either one of the below, note, the model is already trained and stored in model.h5
model.load_model()
# model.train(data.X_train, data.zc_features_train, data.ssc_features_train, data.features_1d_train, data.Y_train)

model.evaluate(data.X_test, data.zc_features_test, data.ssc_features_test, data.features_1d_test, data.Y_test)
# to predict: you need to pass normalized data into model.predict(self, X, zc_features, ssc_features, features_1d)
# to normalize, you can use data.normalize(X, zc_features, ssc_features, features_1d)

## save the model if it achieve a better accuracy
# save = input("Save model?")
# # previous test accuracy: 91
# if int(save) == 1:
#     model.save_model()






