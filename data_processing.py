import pandas as pd
from sklearn import preprocessing


def preprocess_data(data):

    excluded_attributes = ['L3_Src_Addr', 'L3_Dst_Addr', 'flow_id', 'app_encryption']
    data = data.drop(excluded_attributes, axis=1)

    le = preprocessing.LabelEncoder()
    data['encoded'] = le.fit_transform(data['app_id'])

    data = pd.get_dummies(data, columns=["app_package_name"])
    data.fillna(value=0.0, inplace=True)

    return data
