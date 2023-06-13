import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import time

# Задание 2
class Classificator:
    def __init__(self, data):
        self.data = data

    def fit(self):
        train_data_y = self.data['encoded']
        self.data.drop(labels="encoded", axis=1, inplace=True)

        train_data = self.data.values

        state = 12
        test_size = 0.30

        x_train, x_val, y_train, y_val = train_test_split(train_data, train_data_y, test_size=test_size, random_state=state)

        # scaler = MinMaxScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_val = scaler.transform(x_val)

        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)

        # время обучения
        start_time = time.time()
        gb_clf.fit(x_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        print("Training time: {0:.3f} sec.".format(training_time))

        # время предсказания
        start_time = time.time()
        predictions = gb_clf.predict(x_val)
        end_time = time.time()
        prediction_time = end_time - start_time

        print("Prediction time: {0:.3f} sec.".format(prediction_time))

        # gb_clf = RandomForestClassifier(n_estimators=100, max_features=2, max_depth=2, random_state=0)
        # gb_clf.fit(x_train, y_train)

        # gb_clf = GaussianNB()
        # gb_clf.fit(x_train, y_train)

        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(x_val, y_val)))

        print(confusion_matrix(y_val, predictions))

        print("Classification Report")
        print(classification_report(y_val, predictions))
# Задание 2