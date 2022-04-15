#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gc
import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

import optuna
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# In[2]:


df_train1 = pd.read_csv('train1.csv')
df_test1 = pd.read_csv('test1.csv')

df_train2 = pd.read_csv('train2.csv')
df_test2 = pd.read_csv('test2.csv')

df_train3 = pd.read_csv('train3.csv')
df_test3 = pd.read_csv('test3.csv')

df_train4 = pd.read_csv('train4.csv')
df_test4 = pd.read_csv('test4.csv')

# df_train5 = pd.read_csv('train6.csv')
# df_test5 = pd.read_csv('test6.csv')

# df_train = pd.merge(df_train1, df_train2, on=['sn', 'fault_time', 'label'])\
#     .merge(df_train3, on=['sn', 'fault_time', 'label'])\
#     .merge(df_train4, on=['sn', 'fault_time', 'label'])\
#     .merge(df_train5, on=['sn', 'fault_time', 'label'])
# df_test = pd.merge(df_test1, df_test2, on=['sn', 'fault_time'])\
#     .merge(df_test3, on=['sn', 'fault_time'])\
#     .merge(df_test4, on=['sn', 'fault_time'])\
#     .merge(df_test5, on=['sn', 'fault_time'])

df_train = pd.merge(df_train1, df_train2, on=['sn', 'fault_time', 'label'])    .merge(df_train3, on=['sn', 'fault_time', 'label'])    .merge(df_train4, on=['sn', 'fault_time', 'label'])
df_test = pd.merge(df_test1, df_test2, on=['sn', 'fault_time'])    .merge(df_test3, on=['sn', 'fault_time'])    .merge(df_test4, on=['sn', 'fault_time'])


# In[3]:


bert_train = pd.read_csv('../bert/train.csv')
bert_test = pd.read_csv('../bert/test.csv')


# In[4]:


df_train = pd.concat([df_train, bert_train.iloc[:, 9:]], axis=1)
df_test = pd.concat([df_test, bert_test.iloc[:, 9:]], axis=1)


# In[5]:


df_train.shape, df_test.shape


# In[21]:


def macro_f1(y_true, y_pred) -> float:
    """
    计算得分
    :param target_df: [sn,fault_time,label]
    :param submit_df: [sn,fault_time,label]
    :return:
    """
    weights =  [3  /  7,  2  /  7,  1  /  7,  1  /  7]
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


# In[18]:


NUM_CLASSES = df_train['label'].nunique()
FOLDS = 10
TARGET = 'label'
use_features = [col for col in df_train.columns if col not in ['sn', 'fault_time', TARGET]]

def run_ctb(df_train, df_test, use_features):
    target = TARGET
    oof_pred = np.zeros((len(df_train), NUM_CLASSES))
    y_pred = np.zeros((len(df_test), NUM_CLASSES))

    folds = GroupKFold(n_splits=FOLDS)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(df_train, df_train[TARGET], df_train['sn'])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]

        params_ = {
            'objective': 'multi:softmax',
            'eta': 0.03,
            'max_depth': 7,
            'subsample': 0.8,
            'n_estimators': 10000,
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_child_weight': 16,
            'tree_method': 'gpu_hist'
        }
        model = xgb.XGBClassifier(**params_)
        model.fit(x_train, y_train, early_stopping_rounds=100,
                  eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='mlogloss',
                  verbose=100)

        oof_pred[val_ind] = model.predict_proba(x_val)
        y_pred += model.predict_proba(df_test[use_features]) / folds.n_splits

        score = f1_score(y_val, oof_pred[val_ind].argmax(axis=1), average='macro')
        print(f'F1 score: {score}')

        print("Features importance...")
        feat_imp = pd.DataFrame({'imp': model.feature_importances_, 'feature': use_features})
        feat_imp.sort_values(by='imp').to_csv('%d_imp.csv'%fold, index=False)
        print(feat_imp.sort_values(by='imp').reset_index(drop=True))

        del x_train, x_val, y_train, y_val
        gc.collect()

    return y_pred, oof_pred


# In[19]:


y_pred, oof_pred = run_ctb(df_train, df_test, use_features)


# In[22]:


print(macro_f1(df_train[TARGET], np.argmax(oof_pred, axis=1)))


# In[23]:


submit_df = pd.read_csv('../data/preliminary_submit_dataset_a.csv')


# In[24]:


sub = submit_df[['sn', 'fault_time']].copy()
sub['label'] = y_pred.argmax(axis=1)
print(sub.head())
print(sub['label'].value_counts() / sub.shape[0])


# In[25]:


sub.to_csv('baseline3_gkf_sn.csv', index=False)


# In[ ]:


label1 = pd.read_csv('../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

print(label_df['label'].value_counts() / label_df.shape[0])

