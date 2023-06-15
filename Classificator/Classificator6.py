import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import time


class Classificator6:
    def __init__(self, data):
        self.data = data.copy()
        self.clf = None

    def fit(self, pipeline=False, **kwargs):
        """
            model_names:
            ----------
                GradientBoosting
                LogisticRegression
                RandomForest
                KNN
                DecisionTree
                SVM
        """

        train_data_y = self.data['encoded']
        self.data.drop(labels="encoded", axis=1, inplace=True)

        train_data = self.data.values

        state = 12
        test_size = 0.30

        x_train, x_val, y_train, y_val = train_test_split(train_data, train_data_y, test_size=test_size, random_state=state)

        model = RandomForestClassifier(**kwargs)

        if pipeline:
            clf = make_pipeline(StandardScaler(), model)
        else:
            clf = model

        # время обучения
        start_time = time.time()
        clf.fit(x_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        predictions = clf.predict(x_val)

        print("--------------------------------------------")

        print("Training time: {0:.3f} sec.".format(training_time))

        print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(clf.score(x_val, y_val)))

        print(confusion_matrix(y_val, predictions))

        print("Classification Report")
        print(classification_report(y_val, predictions, zero_division=1))

        self.clf = clf
        return clf

    def prediction(self, x_val, y_val):
        print("Accuracy score (validation): {0:.3f}".format(self.clf.score(x_val, y_val)))

        print("--------------------------------------------")

        return self.clf.score(x_val, y_val)
