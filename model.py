from keras.layers import Input, Flatten, Dense, Dropout, LSTM, Bidirectional, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
import os


class BILSTM:
    def train(self, X_train, zc_features_train, ssc_features_train, features_1d_train, Y_train, num_activation=24):
        """"construct model architecture"""
        # input layer setup
        x_branch_input = Input(shape=(X_train.shape[1], X_train.shape[2]))
        zc_branch_input = Input(shape=(zc_features_train.shape[1], zc_features_train.shape[2]))
        ssc_branch_input = Input(shape=(ssc_features_train.shape[1], ssc_features_train.shape[2]))
        oned_branch_input = Input(shape=(features_1d_train.shape[1], features_1d_train.shape[2]))

        # branch setup
        x_branch = Bidirectional(LSTM(num_activation))(x_branch_input)
        x_branch = Dropout(0.5)(x_branch)
        x_branch = Dense(num_activation, activation='relu')(x_branch)
        x_branch = Model(inputs=x_branch_input, outputs=x_branch)

        zc_branch = Bidirectional(LSTM(num_activation))(zc_branch_input)
        zc_branch = Dropout(0.3)(zc_branch)
        zc_branch = Dense(num_activation, activation='relu')(zc_branch)
        zc_branch = Model(inputs=zc_branch_input, outputs=zc_branch)

        ssc_branch = Bidirectional(LSTM(num_activation))(ssc_branch_input)
        ssc_branch = Dropout(0.3)(ssc_branch)
        ssc_branch = Dense(num_activation, activation='relu')(ssc_branch)
        ssc_branch = Model(inputs=ssc_branch_input, outputs=ssc_branch)

        oned_branch = Flatten()(oned_branch_input)
        oned_branch = Dense(num_activation, activation='relu')(oned_branch)
        oned_branch = Model(inputs=oned_branch_input, outputs=oned_branch)

        # root setup
        combined = concatenate([x_branch.output, zc_branch.output, ssc_branch.output, oned_branch.output])
        combined = Dense(num_activation, activation='relu')(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(7, activation='relu')(combined)
        # combined = Dense(num_activation, activation='relu')(combined)
        combined = Dense(7, activation='softmax')(combined)
        # compile model
        self.model = Model(inputs=[x_branch_input, zc_branch_input, ssc_branch_input, oned_branch_input], output=combined)
        opt = Adam(lr=1e-2, decay=1e-2/200)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        # fit model
        self.model.fit([X_train, zc_features_train, ssc_features_train, features_1d_train], Y_train, epochs=12, batch_size=64)
        print(self.model.summary())

    def evaluate(self, X_test, zc_features_test, ssc_features_test, features_1d_test, Y_test):
        self.load_model()
        test_pred = self.get_class(self.model.predict([X_test, zc_features_test, ssc_features_test, features_1d_test]))
        true_pred = self.get_class(Y_test)
        print("test prediction", test_pred)
        print("true prediction", true_pred)
        print(classification_report(true_pred, test_pred))

    def predict(self, X, zc_features, ssc_features, features_1d):
        # precondition: passed in data must be normalized.
        self.load_model()
        return self.get_class(self.model.predict([X, zc_features, ssc_features, features_1d]))

    def get_class(self, predictions):
        """get class label from one hot encoding"""
        result = []
        for prediction in predictions:
            prediction = prediction.tolist()
            result.append(prediction.index(max(prediction)))
        return np.array(result)

    def save_model(self):
        self.model.save("model.h5")

    def load_model(self):
        self.model = load_model('model.h5')
