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

        # выделяем ключи для тренировочной даты
        train_data_y = self.data['encoded']
        self.data.drop(labels="encoded", axis=1, inplace=True)

        # записываем значения колонок всей выборки без названий колонок
        train_data = self.data.values

        state = 12
        # размер тестовой выборки
        test_size = 0.30

        # делим всю выборку на тестовую и нет
        x_train, x_val, y_train, y_val = train_test_split(train_data, train_data_y, test_size=test_size, random_state=state)

        # создаем экземпляр классификатора
        model = RandomForestClassifier(**kwargs)

        if pipeline:
            clf = make_pipeline(StandardScaler(), model)
        else:
            clf = model

        # время обучения
        start_time = time.time()
        # обучаем модель
        clf.fit(x_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        predictions = clf.predict(x_val)

        print("--------------------------------------------")

        print("Training time: {0:.3f} sec.".format(training_time))

        # классифицируем тренировочные данные и тестовые, для сравнения показателей
        print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(clf.score(x_val, y_val)))

        print(confusion_matrix(y_val, predictions))

        print("Classification Report")
        print(classification_report(y_val, predictions, zero_division=1))

        # сохраняем обученный классификатор в класс
        self.clf = clf
        return clf

    def prediction(self, x_val, y_val):
        # создаем фреймы из массивов
        x_val = pd.DataFrame([x_val], columns=['app_id', 'L3_Tot_Pl_Sz_C2S', 'L3_Tot_Pl_Sz_S2C', 'L4_Tot_Pl_Sz_C2S',
                                              'L4_Tot_Pl_Sz_S2C', 'L3_Avg_Dtg_Sz_C2S', 'L3_Avg_Dtg_Sz_S2C',
                                              'L4_Avg_Pl_Sz_C2S', 'L4_Avg_Pl_Sz_C2S.1', 'L3_Std_Tot_Sz_C2S',
                                              'L3_Std_Tot_Sz_S2C', 'L4_Std_Pl_Sz_C2S', 'L4_Std_Pl_Sz_S2C',
                                              'L3_Avg_Pac4Msg_C2S', 'L3_Avg_Pac4Msg_S2C', 'L3_Efficiency_C2S',
                                              'L3_Efficiency_S2C', 'L3_Tot_Dtg_Sz_CS_ratio', 'L4_Tot_Pl_Sz_CS_ratio',
                                              'L3_Tot_Dtg_Cnt_CS_ratio', 'L3_Tot_Dtg_Cnt_C2S', 'L3_Tot_Dtg_Cnt_S2C'])
        y_val_new = pd.DataFrame([y_val], columns=['encoded'])

        # классифицируем данные
        return self.clf.score(x_val.values, y_val_new)
