import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
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

# Вывод выбросов
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


# фильтрация столца app_encryption
data = data[data["app_encryption"] == "yes"]
data_partially = data[data["app_encryption"] == "partially"]


processed_data = preprocess_data(data)
processed_data_partially = preprocess_data(data_partially)

data.to_csv('updated_android_apps_traffic_attributes_prepared.csv', index=False)


classificator = Classificator(processed_data, processed_data_partially)
classificator.fit()
