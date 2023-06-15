import time
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB


class Classificator_unknown:
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

        start_time = time.time()

        sub_data = self.data_partially[:round(len(self.data_partially) / 3)]
        x_val = np.concatenate((x_val, sub_data.drop(columns="encoded").values), axis=0)
        y_val = np.concatenate((y_val, sub_data['encoded']), axis=0)

        unknown_app = sub_data.drop(columns="encoded")
        unknown_mark = np.full(len(unknown_app), "unknown")
        # добавляем в обучающую выборку неизвестное приложение
        x_train = np.concatenate((x_train, unknown_app.values), axis=0)
        y_train = np.concatenate((y_train, unknown_mark), axis=0)

        # преобразование меток в числовой формат
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)

        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
        gb_clf.fit(x_train, y_train_encoded)

        end_time = time.time()
        training_time = end_time - start_time

        start_time = time.time()
        predictions_encoded = gb_clf.predict(x_val)
        end_time = time.time()
        prediction_time = end_time - start_time

        # преобразование меток в исходный формат
        predictions = le.inverse_transform(predictions_encoded)

        print("Training Time: {:.3f} seconds".format(training_time))
        print("Prediction Time: {:.3f} seconds".format(prediction_time))
        print("Accuracy: {:.3f}".format(gb_clf.score(x_val, y_val_encoded)))
        y_val_encoded = y_val_encoded.astype(str)
        print(confusion_matrix(y_val_encoded, predictions))
        print("Classification Report")
        print(classification_report(y_val_encoded, predictions, zero_division=1))