from ast import arg
import os
import pandas as pd
from datetime import datetime
from pycaret.classification import save_model, load_model

def save_model_pkl(model):
    model_name = datetime.now().strftime('%Y-%m-%d-%H-%M-model')
    save_model(model, model_name)
    current_path = os.getcwd()
    if not os.path.exists(f'{current_path}/models'):
        os.mkdir(f'{current_path}/models')
    os.rename(f'{current_path}/{model_name}.pkl', f'{current_path}/models/{model_name}.pkl')


def load_model_pkl():
    current_path = os.getcwd()
    models_path = f'{current_path}/models'
    model_name = get_latest_model()
    return load_model(f'{models_path}/{model_name}')


def get_latest_model():
    current_path = os.getcwd()
    models_path = f'{current_path}/models'
    models = os.listdir(models_path)
    models.sort()
    return models[-1].rsplit('.')[0]


def get_last_auc():
    evals = pd.read_csv('evaluations/model_eval.csv')
    return evals['AUC'].iloc[-1]


def combine_data(batch_num):
    train = pd.read_csv('data/weatherAUS_train.csv')
    test = pd.read_csv('data/weatherAUS_test.csv')
    data = train.append(test, sort=False, ignore_index=True)
    for i in range(1, batch_num):
        batch = pd.read_csv(f'data/prod_batch{i}.csv')
        data = data.append(batch, sort=False, ignore_index=True)

    return data;