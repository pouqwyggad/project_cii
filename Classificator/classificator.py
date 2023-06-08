from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler


class Classificator:
    def __init__(self, data):
        self.data = data

    def fit(self):
        data_length = self.data.count()[0]
        train_data_y = self.data['app_id']
        train_data = self.data.values[0:round(data_length - data_length / 3)]
        test_data = self.data.values[round(data_length - data_length / 3):]

        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        state = 12
        test_size = 0.30

        X_train, X_val, y_train, y_val = train_test_split(train_data, train_data_y,
                                                          test_size=test_size, random_state=state)

        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
        gb_clf.fit(X_train, y_train)

        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

        predictions = gb_clf.predict(X_val)

        print("Classification Report")
        print(classification_report(y_val, predictions))
