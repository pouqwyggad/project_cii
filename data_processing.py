import pandas as pd
from sklearn import preprocessing


def preprocess_data(data):

    # исключаем ненужные атрибуты
    excluded_attributes = ['L3_Src_Addr', 'L3_Dst_Addr', 'flow_id', 'app_encryption', 'app_package_name']
    data = data.drop(excluded_attributes, axis=1)

    # кодируем app_id и сохраняем в колонку новую "encoded"
    le = preprocessing.LabelEncoder()
    data['encoded'] = le.fit_transform(data['app_id'])

    return data
