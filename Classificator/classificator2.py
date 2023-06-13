import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import time

# Задание 3
class Classificator2:
    def __init__(self, data, data_partially):
        self.data = data
        self.data_partially = data_partially

    def fit(self):
        train_data_y = self.data['encoded']
        self.data.drop(labels="encoded", axis=1, inplace=True)

        train_data = self.data.values

        state = 12
        test_size = 0.30

        x_train, x_val, y_train, y_val = train_test_split(train_data, train_data_y, test_size=test_size, random_state=state)

        sub_data = self.data_partially[:round(len(self.data_partially) / 3)]
        x_val = np.concatenate((x_val, sub_data.drop(columns="encoded").values), axis=0)
        y_val = np.concatenate((y_val, sub_data['encoded']), axis=0)

        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)

        start_time = time.time()
        gb_clf.fit(x_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        start_time = time.time()
        predictions = gb_clf.predict(x_val)
        end_time = time.time()
        prediction_time = end_time - start_time

        print("Training time: {0:.3f} sec.".format(training_time))
        print("Prediction time: {0:.3f} sec.".format(prediction_time))
        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(x_val, y_val)))
        print(confusion_matrix(y_val, predictions))
        print("Classification Report")
        print(classification_report(y_val, predictions))
# Задание 3