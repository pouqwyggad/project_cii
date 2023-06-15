import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import time

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Classificator2:
    def __init__(self, data, data_partially):
        self.data = data
        self.data_partially = data_partially

    def fit(self, model_name='GradientBoosting', pipeline=False, **kwargs):
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

        x_val = x_val[:round(len(self.data_partially) / 2)]
        y_val = y_val[:round(len(self.data_partially) / 2)]

        sub_data = self.data_partially[:round(len(x_val) / 2)]

        y_val = np.concatenate((y_val, sub_data['encoded']), axis=0)
        x_val = np.concatenate((x_val, sub_data.drop(columns="encoded").values), axis=0)

        model = None

        if model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(**kwargs)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(**kwargs)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**kwargs)
        elif model_name == 'KNN':
            model = KNeighborsClassifier(**kwargs)
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier(**kwargs)
        elif model_name == 'SVM':
            model = SVC(**kwargs)

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

        print("Model {0}".format(model_name))
        print("--------------------------------------------")

        print("Training time: {0:.3f} sec.".format(training_time))

        print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(clf.score(x_val, y_val)))

        print(confusion_matrix(y_val, predictions))

        print("Classification Report")
        print(classification_report(y_val, predictions, zero_division=1))
        return clf
