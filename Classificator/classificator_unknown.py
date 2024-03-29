import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# мы вводим новый класс "Неизвестное приложение" и повторно обучаем модели
# в обучающей выборке используем partially трафик
# добавляем новый класс "unknown" к целевому признаку

class Classificator_unknown:
    def __init__(self, data, data_partially):
        self.data = data.copy()
        self.data_partially = data_partially.copy()

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

        # выделяем ключи для тренировочной даты
        train_data_y = self.data['encoded']
        self.data.drop(labels="encoded", axis=1, inplace=True)

        # записываем значения колонок всей выборки без названий колонок
        train_data = self.data.values

        state = 12
        # размер тестовой выборки
        test_size = 0.30

        # разделение данных на обучающую и тестовую выборку
        x_train, x_val, y_train, y_val = train_test_split(train_data, train_data_y, test_size=test_size, random_state=state)

        # запускаем таймер
        start_time = time.time()

        # берем треть от неизвестных приложений
        sub_data = self.data_partially[:round(len(self.data_partially) / 3)]

        # заменяем часть тестовой выборки неизвестными приложениями
        x_val = np.concatenate((x_val[:round(len(self.data_partially) / 3)], sub_data.drop(columns="encoded").values), axis=0)
        y_val = np.concatenate((y_val[:round(len(self.data_partially) / 3)], sub_data['encoded']), axis=0)


        # используем это для обучения модели на неизвестных данных
        # вводим новвый класс "Неизвестное приложение" в обучающую выборку
        unknown_app = sub_data.drop(columns="encoded")
        unknown_mark = np.full(len(unknown_app), "unknown")
        x_train = np.concatenate((x_train, unknown_app.values), axis=0)
        y_train = np.concatenate((y_train, unknown_mark), axis=0)

        # преобразование меток в числовой формат
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)

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
        clf.fit(x_train, y_train_encoded)
        end_time = time.time()
        training_time = end_time - start_time

        start_time = time.time()
        predictions_encoded = clf.predict(x_val)
        end_time = time.time()
        prediction_time = end_time - start_time

        # преобразуем предсказанные метки обратно в исходный формат
        predictions = le.inverse_transform(predictions_encoded)

        print("Model {0}".format(model_name))
        print("--------------------------------------------")

        print("Training time: {0:.3f} sec.".format(training_time))
        print("Prediction time: {0:.3f} sec.".format(prediction_time))

        # классифицируем тренировочные данные и тестовые, для сравнения показателей
        print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train_encoded)))
        print("Accuracy score (validation): {0:.3f}".format(clf.score(x_val, y_val_encoded)))

        # преобразуем метки классов в один тип (str)
        y_val_encoded = y_val_encoded.astype(str)
        print(confusion_matrix(y_val_encoded, predictions))

        print("Classification Report")
        print(classification_report(y_val_encoded, predictions, zero_division=1))
        return clf
