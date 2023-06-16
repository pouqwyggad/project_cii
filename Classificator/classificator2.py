import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
import time

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# взяли половину тестовой выборки из предыдущего задания
# добавили к этой выборке приложения частично шифрующие трафик "data_partially"
# рассчитали метрики
# замерили время предсказания модели
class Classificator2:
    def __init__(self, data, data_partially): # сюда передаем data отсортированное по "yes" и "partially"
        self.data = data.copy() # копируем данные
        self.data_partially = data_partially.copy() # копируем данные

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

        train_data_y = self.data['encoded'] # целевая переменная которую мы пытаемся предсказать
        self.data.drop(labels="encoded", axis=1, inplace=True)

        train_data = self.data.values

        state = 12
        # размер тестовой выборки
        test_size = 0.30

        # разделение данных на обучающую и тестовую выборку
        x_train, x_val, y_train, y_val = train_test_split(train_data, train_data_y, test_size=test_size, random_state=state)

        # x_val и y_val уменьшаются до половины их исходного размера

        # x_val уменьшается до половины размера data_partially
        x_val = x_val[:round(len(self.data_partially) / 2)]
        # то же самое
        y_val = y_val[:round(len(self.data_partially) / 2)]

        # первая половина data_partially
        sub_data = self.data_partially[:round(len(x_val) / 2)]

        # y_val и x_val расширяются, добавляя метки классов и данные из sub_data
        # метки классов добавляются к y_val
        y_val = np.concatenate((y_val, sub_data['encoded']), axis=0)
        # данные из sub_data добавляются к x_val
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

        # классифицируем тренировочные данные и тестовые, для сравнения показателей
        print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(clf.score(x_val, y_val)))

        print(confusion_matrix(y_val, predictions))

        print("Classification Report")
        print(classification_report(y_val, predictions, zero_division=1))
        return clf
