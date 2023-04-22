import pandas as pd
import os
from datetime import datetime

def log_auc(auc, version):
    eval_df = pd.DataFrame()
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    eval_df = eval_df.append({'time_stamp':now, 'version': version, 'AUC': auc}, ignore_index=True)

    current_path = os.getcwd()
    evaluation_file_name = 'evaluations/model_eval.csv'

    if os.path.isfile(f'{current_path}/{evaluation_file_name}'):
        eval_df.to_csv(f'{current_path}/{evaluation_file_name}', mode='a', index=False, header=False)
    else:
        eval_df.to_csv(f'{current_path}/{evaluation_file_name}', index=False)


def is_drifting(last_auc, new_auc):
    if last_auc > new_auc:
        return True
    else:
        return False