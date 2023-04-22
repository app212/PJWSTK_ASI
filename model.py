from pycaret.classification import setup, compare_models, tune_model, automl, predict_model, pull
from utils import save_model_pkl, load_model_pkl, get_last_auc, combine_data
from monitor import log_auc, is_drifting
from data import prep_data, prep_retrain_data

def train(data):
    exp1 = setup(data=data, target='RainTomorrow', session_id=123,
                normalize=True,
                remove_outliers=True,
                silent=True,
                log_data=False,
                log_experiment=False,
                log_profile=False)
    top3 = compare_models(n_select=3, include=['lr', 'nb', 'rf', 'lda'], cross_validation=False) 
    # tune models
    tuned_top3 = [tune_model(i, n_iter=1, choose_better=True) for i in top3]

    # get best model from current setup
    model = automl(optimize = 'AUC')
    # save model to pickle
    save_model_pkl(model)


def evaluate(data):
    model = load_model_pkl()
    predict_model(model, data=data)

    metrics = pull()
    log_auc(metrics['AUC'].iloc[0], 'evaluation')


def prod(batch_num):
    iter = 0

    if batch_num > 5: 
        batch_num = 5

    while iter <  batch_num:
        model = load_model_pkl()
        data = prep_data(f'data/prod_batch{iter+1}.csv')
        predict_model(model, data=data)

        metrics = pull()
        new_auc = metrics['AUC'].iloc[0]
        last_auc = get_last_auc()

        if is_drifting(last_auc, new_auc):
            retrain(iter+1)
        else:
            log_auc(metrics['AUC'].iloc[0], f'prod_batch{iter+1}')
        
        iter += 1


def retrain(batch_num):
    data = combine_data(batch_num)
    preped_data = prep_retrain_data(data)

    train(preped_data)
    metrics = pull()
    log_auc(metrics['AUC'].iloc[0], f'retrain_batch{batch_num}')