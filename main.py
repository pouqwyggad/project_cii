import pandas as pd
import matplotlib.pyplot as plt

from Classificator.classificator2 import Classificator2
from Classificator.Classificator6 import Classificator6
from Classificator.classificator_unknown import Classificator_unknown
from data_processing import preprocess_data

from Classificator.classificator import Classificator

print("Задание 1:")
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
for column in numeric_columns:
    plt.figure(figsize=(6, 4))
    plt.boxplot(data[column])
    plt.title(column)
    plt.close()

    plt.savefig(column + ".png")
    plt.show()

print("Задание 2:")

# берем из всей выборки только зашифрованные
data_yes = data[data["app_encryption"] == "yes"]
# берем из всей выборки только не зашифрованные
data_no = data[data["app_encryption"] == "no"]
# берем из всей выборки с плавающим шифрованием
data_partially = data[data["app_encryption"] == "partially"]

# удаляем все лишнее
processed_data_yes = preprocess_data(data_yes)
processed_data_no = preprocess_data(data_no)
processed_data_partially = preprocess_data(data_partially)

# создаем экземпляры классификатора обученного на зашифрованных приложениях
classificator_yes_1 = Classificator(processed_data_yes)
classificator_yes_2 = Classificator(processed_data_yes)
classificator_yes_3 = Classificator(processed_data_yes)
classificator_yes_4 = Classificator(processed_data_yes)
classificator_yes_5 = Classificator(processed_data_yes)
classificator_yes_6 = Classificator(processed_data_yes)

# создаем экземпляры классификатора обученного на не зашифрованных приложениях
classificator_no_1 = Classificator(processed_data_no)
classificator_no_2 = Classificator(processed_data_no)
classificator_no_3 = Classificator(processed_data_no)
classificator_no_4 = Classificator(processed_data_no)
classificator_no_5 = Classificator(processed_data_no)
classificator_no_6 = Classificator(processed_data_no)

# обучаем
print('Encripted YES')
clf_yes_1 = classificator_yes_1.fit(model_name='GradientBoosting')
clf_yes_2 = classificator_yes_2.fit(model_name='LogisticRegression')
clf_yes_3 = classificator_yes_3.fit(model_name='RandomForest')
clf_yes_4 = classificator_yes_4.fit(model_name='KNN')
clf_yes_5 = classificator_yes_5.fit(model_name='DecisionTree')
clf_yes_6 = classificator_yes_6.fit(model_name='SVM')

# обучаем
print('Encripted NO')
clf_no_1 = classificator_no_1.fit(model_name='GradientBoosting')
clf_no_2 = classificator_no_2.fit(model_name='LogisticRegression')
clf_no_3 = classificator_no_3.fit(model_name='RandomForest')
clf_no_4 = classificator_no_4.fit(model_name='KNN')
clf_no_5 = classificator_no_5.fit(model_name='DecisionTree')
clf_no_6 = classificator_no_6.fit(model_name='SVM')

# создаем экземпляры классификатора обученного на зашифрованных приложениях
# с добавлением в тестовую выборку неизвесных приложений
# print("Задание 3: ")
clf_3_1 = Classificator2(processed_data_yes, processed_data_partially)
clf_3_2 = Classificator2(processed_data_yes, processed_data_partially)
clf_3_3 = Classificator2(processed_data_yes, processed_data_partially)
clf_3_4 = Classificator2(processed_data_yes, processed_data_partially)
clf_3_5 = Classificator2(processed_data_yes, processed_data_partially)
clf_3_6 = Classificator2(processed_data_yes, processed_data_partially)

# обучаем
clf_3_1.fit(model_name='GradientBoosting')
clf_3_2.fit(model_name='LogisticRegression')
clf_3_3.fit(model_name='RandomForest')
clf_3_4.fit(model_name='KNN')
clf_3_5.fit(model_name='DecisionTree')
clf_3_6.fit(model_name='SVM')

print("Задание 4:")

# создаем экземпляры классификатора обученного на зашифрованных приложениях
# с добавлением в тестовую выборку неизвесных приложений
classificator_partially_1 = Classificator_unknown(processed_data_yes, processed_data_partially)
classificator_partially_2 = Classificator_unknown(processed_data_yes, processed_data_partially)
classificator_partially_3 = Classificator_unknown(processed_data_yes, processed_data_partially)
classificator_partially_4 = Classificator_unknown(processed_data_yes, processed_data_partially)
classificator_partially_5 = Classificator_unknown(processed_data_yes, processed_data_partially)
classificator_partially_6 = Classificator_unknown(processed_data_yes, processed_data_partially)

clf_partially_1 = classificator_partially_1.fit(model_name='GradientBoosting')
clf_partially_2 = classificator_partially_2.fit(model_name='LogisticRegression')
clf_partially_3 = classificator_partially_3.fit(model_name='RandomForest')
clf_partially_4 = classificator_partially_4.fit(model_name='KNN')
clf_partially_5 = classificator_partially_5.fit(model_name='DecisionTree')
clf_partially_6 = classificator_partially_6.fit(model_name='SVM')

data_all = data.copy()
data_all = preprocess_data(data_all)

# выделяем из выборки отдельные приложения с шифрованием
app_1 = data_all[data_all["app_id"] == 3]
app_2 = data_all[data_all["app_id"] == 9]
app_3 = data_all[data_all["app_id"] == 10]
app_4 = data_all[data_all["app_id"] == 12]
app_5 = data_all[data_all["app_id"] == 14]
app_6 = data_all[data_all["app_id"] == 17]

# выделяем из выборки отдельные приложения без шифрования
app_7 = data_all[data_all["app_id"] == 13]
app_8 = data_all[data_all["app_id"] == 23]
app_9 = data_all[data_all["app_id"] == 48]
app_10 = data_all[data_all["app_id"] == 61]
app_11 = data_all[data_all["app_id"] == 58]
app_12 = data_all[data_all["app_id"] == 82]

app_5.to_csv('test')
app_10.to_csv('test2')

# создаем экземпляр класса
clf_6_1 = Classificator6(app_1)
clf_6_2 = Classificator6(app_2)
clf_6_3 = Classificator6(app_3)
clf_6_4 = Classificator6(app_4)
clf_6_5 = Classificator6(app_5)
clf_6_6 = Classificator6(app_6)
clf_6_7 = Classificator6(app_7)
clf_6_8 = Classificator6(app_8)
clf_6_9 = Classificator6(app_9)
clf_6_10 = Classificator6(app_10)
clf_6_11 = Classificator6(app_11)
clf_6_12 = Classificator6(app_12)

# обучаем
clf_6_1.fit()
clf_6_2.fit()
clf_6_3.fit()
clf_6_4.fit()
clf_6_5.fit()
clf_6_6.fit()
clf_6_7.fit()
clf_6_8.fit()
clf_6_9.fit()
clf_6_10.fit()
clf_6_11.fit()
clf_6_12.fit()

# добавляем обученные классификаторы в массив
classificators = [clf_6_1, clf_6_2, clf_6_3, clf_6_4, clf_6_5, clf_6_6, clf_6_7, clf_6_8, clf_6_9, clf_6_10, clf_6_11, clf_6_12]

# объединяем все данные в один массив

# достаем от туда колонку encoded
y_val = data_all['encoded']
y_val = y_val.values

# удаляем из общей выборки колонку encoded
data_all.drop(labels="encoded", axis=1, inplace=True)
data_all.to_csv('test')
data_all = data_all.values

i = 0
# запускаем цикл по всей выборке, на каждой итерации достается строка с данными по колонкам в виде массива
for line in data_all:
    max = -1
    app_id = 0
    # запускаем цикл по всем классификаторам
    for j, classificator in enumerate(classificators):
        # классифицируем
        prediction = classificator.prediction(x_val=line, y_val=y_val[i])

        # если есть совпадение, и оно больше максимального, перезаписываем max и созраняем id приложения
        if prediction > max:
            max = prediction
            app_id = line[0]

    if max > 0.5:
        print("App: {0}".format(round(app_id)))
    else:
        print("App: unknown")

    i += 1