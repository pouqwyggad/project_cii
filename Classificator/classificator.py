from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB


class Classificator:
    def __init__(self, data):
        self.data = data

    def fit(self):
        train_data_y = self.data['encoded']
        self.data.drop(labels="encoded", axis=1, inplace=True)

        train_data = self.data.values

        state = 12
        test_size = 0.30

        X_train, X_val, y_train, y_val = train_test_split(train_data, train_data_y,
                                                          test_size=test_size, random_state=state)

        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_val = scaler.transform(X_val)

        gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
        gb_clf.fit(X_train, y_train)

        # gb_clf = RandomForestClassifier(n_estimators=100, max_features=2, max_depth=2, random_state=0)
        # gb_clf.fit(X_train, y_train)

        # gb_clf = GaussianNB()
        # gb_clf.fit(X_train, y_train)

        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))

        predictions = gb_clf.predict(X_val)

        print(confusion_matrix(y_val, predictions))

        print("Classification Report")
        print(classification_report(y_val, predictions))
