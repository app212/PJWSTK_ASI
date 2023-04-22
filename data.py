import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

def prep_data(data_file):
    data = pd.read_csv(data_file)

    # --Data cleanup --

    # parse Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)

    # fill missing values with mode
    for i in object_cols:
        data[i].fillna(data[i].mode()[0], inplace=True)

    # list of numeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    # fill missing values with median
    for i in num_cols:
        data[i].fillna(data[i].median(), inplace=True)

    #-- Data preprocessing --

    # encode categorical features
    label_encoder = LabelEncoder()

    for i in object_cols:
        data[i] = label_encoder.fit_transform(data[i])

    # scale features
    features = data.drop(['RainTomorrow', 'Date'], axis=1)
    target = data['RainTomorrow']

    col_names = list(features.columns)
    s_scaler = preprocessing.StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names)
    return data

def prep_retrain_data(data):
    # --Data cleanup --
    # parse Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # list of categorical variables
    s = (data.dtypes == "object")
    object_cols = list(s[s].index)

    # fill missing values with mode
    for i in object_cols:
        data[i].fillna(data[i].mode()[0], inplace=True)

    # list of numeric variables
    t = (data.dtypes == "float64")
    num_cols = list(t[t].index)

    # fill missing values with median
    for i in num_cols:
        data[i].fillna(data[i].median(), inplace=True)

    #-- Data preprocessing --

    # encode categorical features
    label_encoder = LabelEncoder()

    for i in object_cols:
        data[i] = label_encoder.fit_transform(data[i])

    # scale features
    features = data.drop(['RainTomorrow', 'Date'], axis=1)
    target = data['RainTomorrow']

    col_names = list(features.columns)
    s_scaler = preprocessing.StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names)
    return data