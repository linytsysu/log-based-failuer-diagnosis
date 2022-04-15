import numpy as np
import pandas as pd
from age_model import get_age_prediction

data_path = '../data/'
df_tr               = pd.read_csv(data_path + 'train.csv')
df_tr_app_events    = pd.read_csv(data_path + 'train_app_events.csv')
df_te               = pd.read_csv(data_path + 'test.csv')
df_te_app_events    = pd.read_csv(data_path + 'test_app_events.csv')

df_tr_app_events = df_tr_app_events.merge(df_tr, on='device_id')

X_train = train_df = pd.read_csv('./feature_data.csv')
X_test = pd.read_csv('./feature_test_data.csv')

y_train = df_tr_app_events.groupby(['device_id', 'age']).size().reset_index()['age'].values

age_val_score, age_pred = get_age_prediction(X_train, y_train, X_test)