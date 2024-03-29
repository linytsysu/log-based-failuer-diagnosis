import os
import gc
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
from tqdm import tqdm
from imblearn.over_sampling import SMOTENC


dtype = {
    'server_model': 'str',
    'last_msg_id': 'str',
    'last_template_id': 'str',
    'tmp_appearance_1': 'str',
    'tmp_appearance_2': 'str',
    'tmp_appearance_3': 'str',
    'msg_appearance_1': 'str',
    'msg_appearance_2': 'str',
    'msg_appearance_3': 'str',
    'max_continuous_msg': 'str'
}
df_train = pd.read_csv('../user_data/train.csv', dtype=dtype)
df_test = pd.read_csv('../user_data/test.csv', dtype=dtype)

for name in ['server_model', 'last_msg_id', 'last_template_id', 'tmp_appearance_1', 'tmp_appearance_2',
             'tmp_appearance_3', 'msg_appearance_1', 'msg_appearance_2', 'msg_appearance_3', 'max_continuous_msg']:
    df_train[name] = df_train[name].fillna('MISSING')
    df_test[name] = df_test[name].fillna('MISSING')

df_train = df_train.fillna(df_train.mean(numeric_only=True))
df_test = df_test.fillna(df_test.mean(numeric_only=True))

print(df_train.shape, df_test.shape)

use_features = [col for col in df_train.columns if col not in ['sn', 'fault_time', 'label']]

features_train, labels_train = df_train[use_features], df_train['label']

cat_features = ['server_model', 'last_msg_id', 'last_template_id',
                'tmp_appearance_1', 'tmp_appearance_2', 'tmp_appearance_3',
                'msg_appearance_1', 'msg_appearance_2', 'msg_appearance_3',
                'max_continuous_msg']
cat_idxs = []
for i in range(len(use_features)):
    if use_features[i] in cat_features:
        cat_idxs.append(i)

oversampler = SMOTENC(random_state=0, categorical_features=cat_idxs)
os_features,os_labels = oversampler.fit_resample(features_train, labels_train)
os_features['label'] = os_labels.values.tolist()

df_train = os_features.copy()


def macro_f1(y_true, y_pred) -> float:
    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """
    weights =  [5  /  11,  4  /  11,  1  /  11,  1  /  11]
    overall_df = pd.DataFrame([y_true, y_pred]).T
    overall_df.columns = ['label_gt', 'label_pr']

    macro_F1 =  0.
    for i in  range(len(weights)):
        TP =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] == i)])
        FP =  len(overall_df[(overall_df['label_gt'] != i) & (overall_df['label_pr'] == i)])
        FN =  len(overall_df[(overall_df['label_gt'] == i) & (overall_df['label_pr'] != i)])
        precision = TP /  (TP + FP)  if  (TP + FP)  >  0  else  0
        recall = TP /  (TP + FN)  if  (TP + FP)  >  0  else  0
        F1 =  2  * precision * recall /  (precision + recall)  if  (precision + recall)  >  0  else  0
        macro_F1 += weights[i]  * F1
    return macro_F1

def cat_model_train(x_train, y_train, x_val, y_val, mode='multiclass'):
    NUM_CLASSES = y_train.nunique()
    # class_weights = {0: 5/7, 1: 1/7, 2: 1/7} if mode == 'multiclass' else {0: 3/7, 1: 2/7}
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    # print(NUM_CLASSES, class_weights)
    params = {
        'task_type': 'CPU',
        'bootstrap_type': 'Bayesian',
        'boosting_type': 'Plain',
        'learning_rate': 0.03 if mode == 'multiclass' else 0.01,
        'eval_metric': 'MultiClass',
        'loss_function': 'MultiClass',
        'classes_count': NUM_CLASSES,
        'iterations': 10000,
        'random_state': 42,
        'depth': 8,
        'leaf_estimation_iterations': 8,
        'reg_lambda': 5,
        # 'subsample': 0.8,
        'class_weights': class_weights,
        'early_stopping_rounds': 100,
        'cat_features': ['server_model', 'last_msg_id', 'last_template_id',
                         'tmp_appearance_1', 'tmp_appearance_2', 'tmp_appearance_3',
                         'msg_appearance_1', 'msg_appearance_2', 'msg_appearance_3',
                         'max_continuous_msg'],
    }
    model = CatBoostClassifier(**params)
    model.fit(x_train,
               y_train,
               eval_set=(x_val, y_val),
               verbose=False)
    return model

print(df_train.shape, df_test.shape)

FOLDS = 10
target = 'label'
use_features = [col for col in df_train.columns if col not in ['sn', 'fault_time', target]]
oof_pred = np.zeros((len(df_train),))

y_pred1 = np.zeros((len(df_test), 3))
y_pred2 = np.zeros((len(df_test), 2))

folds = GroupKFold(n_splits=FOLDS)
for fold, (tr_ind, val_ind) in enumerate(folds.split(df_train, df_train['label'], df_train['server_model'])):
    print('fold: %d'%fold)
    df_trian_sub = df_train.iloc[tr_ind].copy()
    df_valid_sub = df_train.iloc[val_ind].copy()

    x_train1, x_val1 = df_trian_sub[use_features], df_valid_sub[use_features]
    y_train1, y_val1 = df_trian_sub[target], df_valid_sub[target]
    y_train1 = y_train1.apply(lambda x: 0 if x <= 1 else x-1)
    y_val1 = y_val1.apply(lambda x: 0 if x <= 1 else x-1)
    model1 = cat_model_train(x_train1, y_train1, x_val1, y_val1)

    # print("Features importance...")
    # feat_imp = pd.DataFrame({'imp': model1.feature_importances_, 'feature': use_features})
    # feat_imp.sort_values(by='imp').to_csv('%d_imp1.csv'%fold, index=False)
    # print(feat_imp.sort_values(by='imp').reset_index(drop=True))

    trn_proba = model1.predict_proba(x_train1)
    val_proba = model1.predict_proba(x_val1)

    df_trian_sub['proba_0'] = trn_proba[:, 0]
    df_trian_sub['proba_1'] = trn_proba[:, 1]
    df_trian_sub['proba_2'] = trn_proba[:, 2]
    df_valid_sub['proba_0'] = val_proba[:, 0]
    df_valid_sub['proba_1'] = val_proba[:, 1]
    df_valid_sub['proba_2'] = val_proba[:, 2]

    x_train2, x_val2 = df_trian_sub[df_trian_sub['label'] <= 1][use_features + ['proba_0', 'proba_1', 'proba_2']], \
                       df_valid_sub[df_valid_sub['label'] <= 1][use_features + ['proba_0', 'proba_1', 'proba_2']]
    y_train2, y_val2 = df_trian_sub[df_trian_sub['label'] <= 1][target],\
                       df_valid_sub[df_valid_sub['label'] <= 1][target]
    model2 = cat_model_train(x_train2, y_train2, x_val2, y_val2, mode='binary')

    # print("Features importance...")
    # feat_imp = pd.DataFrame({'imp': model2.feature_importances_, 'feature': use_features + ['proba_0', 'proba_1', 'proba_2']})
    # feat_imp.sort_values(by='imp').to_csv('%d_imp2.csv'%fold, index=False)
    # print(feat_imp.sort_values(by='imp').reset_index(drop=True))

    y_pred1 += model1.predict_proba(df_test[use_features]) / folds.n_splits
    df_test['proba_0'] = y_pred1[:, 0]
    df_test['proba_1'] = y_pred1[:, 1]
    df_test['proba_2'] = y_pred1[:, 2]
    y_pred2 += model2.predict_proba(df_test[use_features + ['proba_0', 'proba_1', 'proba_2']]) / folds.n_splits

    val_pred = []
    val_proba = model1.predict_proba(x_val1)
    for i in range(val_proba.shape[0]):
        if np.argmax(val_proba[i]) == 0:
            proba = model2.predict_proba([x_val1.iloc[i].values.tolist() + val_proba[i].tolist()])
            val_pred.append(np.argmax(proba[0]))
        else:
            val_pred.append(np.argmax(val_proba[i])+1)

    score = f1_score(df_valid_sub[target], val_pred, average='macro')
    print(f'F1 score: {score}')
    score = macro_f1(df_valid_sub[target].values, val_pred)
    print(f'F1 score: {score}')

    oof_pred[val_ind] = val_pred

from sklearn.metrics import classification_report
print(classification_report(df_train[target], oof_pred))

print(macro_f1(df_train[target].values, oof_pred))
