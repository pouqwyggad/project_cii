import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

from Classificator.classificator2 import Classificator2
from Classificator.classificator_unknown import Classificator_unknown
from data_processing import preprocess_data

from Classificator.classificator import Classificator


data = pd.read_csv('android_apps_traffic_attributes_prepared.csv')

miss_values = data.isnull().sum()
print("Проверка пропущенных значений по столбцам:")
print(miss_values)

data = data.dropna()  # удаление строк с пропущенными значениями

duplicates = data.duplicated()
print("Проверка наличия дубликатов:")
print(duplicates.sum())

data = data.drop_duplicates()  # удаляем дубликаты, если они есть

data_types = data.dtypes
print("Анализ типов данных:")
print(data_types)

numeric_features = data.select_dtypes(include=['float64', 'int64'])
print("Определение числовых признаков:")
print(numeric_features.columns)

categorical_features = data.select_dtypes(include=['object'])
print("Определение категориальных признаков:")
print(categorical_features.columns)

encoded = pd.get_dummies(data, columns=categorical_features.columns)

numeric_columns = numeric_features.columns
outliers = pd.DataFrame(columns=numeric_columns)

# Проверка наличия выбросов в каждом числовом признаке
for column in numeric_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Определение границ выбросов
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Фильтрация выбросов
    column_outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]

    outliers = pd.concat([outliers, column_outliers], axis=1)

print("Выбросы в числовых признаках:")
print(outliers)

# Создание диаграммы размаха для каждого числового признака
# for column in numeric_columns:
    # plt.figure(figsize=(6, 4))
    # plt.boxplot(data[column])
    # plt.title(column)
    # plt.close()

    # plt.savefig(column + ".png")
    # plt.show()

data_yes = data[data["app_encryption"] == "yes"]
data_no = data[data["app_encryption"] == "no"]
data_partially = data[data["app_encryption"] == "partially"]

processed_data_yes = preprocess_data(data_yes)
processed_data_no = preprocess_data(data_no)
processed_data_partially = preprocess_data(data_partially)

classificator_yes_1 = Classificator(processed_data_yes)
classificator_yes_2 = Classificator(processed_data_yes)
classificator_yes_3 = Classificator(processed_data_yes)
classificator_yes_4 = Classificator(processed_data_yes)
classificator_yes_5 = Classificator(processed_data_yes)
classificator_yes_6 = Classificator(processed_data_yes)
classificator_no_1 = Classificator(processed_data_no)
classificator_no_2 = Classificator(processed_data_no)
classificator_no_3 = Classificator(processed_data_no)
classificator_no_4 = Classificator(processed_data_no)
classificator_no_5 = Classificator(processed_data_no)
classificator_no_6 = Classificator(processed_data_no)
classificator_partially_1 = Classificator(processed_data_partially)
classificator_partially_2 = Classificator(processed_data_partially)
classificator_partially_3 = Classificator(processed_data_partially)
classificator_partially_4 = Classificator(processed_data_partially)
classificator_partially_5 = Classificator(processed_data_partially)
classificator_partially_6 = Classificator(processed_data_partially)


clf_1 = classificator_yes_1.fit(model_name='GradientBoosting')
clf_2 = classificator_yes_2.fit(model_name='LogisticRegression')
clf_3 = classificator_yes_3.fit(model_name='RandomForest')
clf_4 = classificator_yes_4.fit(model_name='KNN')
clf_5 = classificator_yes_5.fit(model_name='DecisionTree')
clf_6 = classificator_yes_6.fit(model_name='SVM')

clf_no_1 = classificator_no_1.fit(model_name='GradientBoosting')
clf_no_2 = classificator_no_2.fit(model_name='LogisticRegression')
clf_no_3 = classificator_no_3.fit(model_name='RandomForest')
clf_no_4 = classificator_no_4.fit(model_name='KNN')
clf_no_5 = classificator_no_5.fit(model_name='DecisionTree')
clf_no_6 = classificator_no_6.fit(model_name='SVM')

clf_partially_1 = classificator_partially_1.fit(model_name='GradientBoosting')
clf_partially_2 = classificator_partially_2.fit(model_name='LogisticRegression')
clf_partially_3 = classificator_partially_3.fit(model_name='RandomForest')
clf_partially_4 = classificator_partially_4.fit(model_name='KNN')
clf_partially_5 = classificator_partially_5.fit(model_name='DecisionTree')
clf_partially_6 = classificator_partially_6.fit(model_name='SVM')

# classificator_unknown = Classificator_unknown(processed_data, processed_data_partially)
# classificator_unknown.fit()

