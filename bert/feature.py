import os
os.environ['TF_KERAS'] = '1'

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from bert4keras.backend import keras
from bert4keras.optimizers import Adam
from bert4keras.models import build_transformer_model
from tqdm import tqdm

from dataset import data, test_data, test_data_b, tokens, keep_tokens
from model import config_path, checkpoint_path, masked_crossentropy, adversarial_training
from data_generator import data_generator, sample_convert


# data_path = '../data/'
# df_tr               = pd.read_csv(data_path + 'train.csv')
# df_tr_app_events    = pd.read_csv(data_path + 'train_app_events.csv')
# df_te               = pd.read_csv(data_path + 'test.csv')
# df_te_app_events    = pd.read_csv(data_path + 'test_app_events.csv')

# df_tr_app_events = df_tr_app_events.merge(df_tr, on='device_id')
# df_te_app_events = df_te_app_events.merge(df_te, on='device_id')

# bert_feature = pd.read_csv('feature_data.csv')
# bert_feature['device_id'] = df_tr_app_events.groupby(['device_id', 'age']).size().reset_index()['device_id'].values
# bert_feature['age'] = df_tr_app_events.groupby(['device_id', 'age']).size().reset_index()['age'].values

# bert_feature_test = pd.read_csv('feature_test_data.csv')
# bert_feature_test['device_id'] = df_te_app_events.groupby('device_id').size().reset_index()['device_id'].values

# age_bert_feature = bert_feature.groupby('age').mean().reset_index()

# age_bert_tr = []
# for x in bert_feature.values:
#     age_bert_tr.append([(x[:-2] * item[1:-1] * (1/np.linalg.norm(x[:-2])) * (1/np.linalg.norm(item[1:-1]))).sum() for item in age_bert_feature.values])
# age_bert_tr = pd.DataFrame(age_bert_tr, columns=['bert_%d'%age for age in age_bert_feature['age'].values])
# age_bert_tr['device_id'] = df_tr_app_events.groupby(['device_id', 'age']).size().reset_index()['device_id'].values
# age_bert_tr.to_csv('age_bert_train.csv', index=False)

# age_bert_te = []
# for x in bert_feature_test.values:
#     age_bert_te.append([(x[:-1] * item[1:-1] * (1/np.linalg.norm(x[:-1])) * (1/np.linalg.norm(item[1:-1]))).sum() for item in age_bert_feature.values])
# age_bert_te = pd.DataFrame(age_bert_te, columns=['bert_%d'%age for age in age_bert_feature['age'].values])
# age_bert_te['device_id'] = df_te_app_events.groupby('device_id').size().reset_index()['device_id'].values
# age_bert_te.to_csv('age_bert_test.csv', index=False)

n_splits = 5
for fold in range(n_splits):
    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='nezha',
        with_mlm=True,
        keep_tokens=[0, 100, 101, 102, 103, 100, 100, 100, 100] + keep_tokens[:len(tokens)]
    )
    model.compile(loss=masked_crossentropy, optimizer=Adam(1e-5))
    model.load_weights('../user_data/best_model_reverse_flod%d.h5'%fold)

    train_generator = data_generator(data, 64)
    feature_data = []
    for x_true, y_true in train_generator:
        feature_data += np.mean(model.predict(x_true), axis=1).tolist()

    test_generator = data_generator(test_data_b, 64)
    feature_test_data = []
    for x_true, _ in test_generator:
        feature_test_data += np.mean(model.predict(x_true), axis=1).tolist()

    feature_data = pd.DataFrame(feature_data)
    feature_data.to_csv('feature_data_fold%d.csv'%fold, index=0)

    feature_test_data = pd.DataFrame(feature_test_data)
    feature_test_data.to_csv('feature_test_data_b_fold%d.csv'%fold, index=0)
