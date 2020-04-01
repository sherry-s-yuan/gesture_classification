import numpy as np
from model import BILSTM
import pandas as pd
from preprocessing import Data


# taking the first 0.5 second of the signal data just as an example
# note each 1800 time is 0.5 second
# signal should have shape (1, 1800, 8)
signal = np.array([pd.read_csv("1_raw_data_13-12_22.03.16.txt", delimiter='\t').to_numpy()[1800:3600, 1:-1]])
print(signal.shape)
# Prepare Data
data = Data(signal)
# initialize model
model = BILSTM()
# load model
model.load_model()
# return prediction, the prediction for time 0.5-1 second is 1 (which is indeed true)
print(model.predict(data.X, data.zc_feature, data.ssc_feature, data.feature_1d))






