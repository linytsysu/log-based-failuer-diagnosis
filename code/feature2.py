import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from scipy.stats import skew, kurtosis
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage")
args = parser.parse_args()
stage = args.stage

label1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_train_label_dataset.csv'))
label2 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_train_label_dataset_s.csv'))
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

if stage == 'final_a':
    submit_df = pd.read_csv('/tcdata/final_submit_dataset_a.csv')
elif stage == 'final_b':
    submit_df = pd.read_csv('/tcdata/final_submit_dataset_b.csv')
else:
    submit_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_submit_dataset_b.csv'))

print(label_df.shape, submit_df.shape)

log_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../user_data/log_template.csv'))
log_df['msg_lower'] = log_df['msg_lower'].astype(str)
log_df['server_model'] = log_df['server_model'].astype(str)

log_df['time'] = pd.to_datetime(log_df['time'])
label_df['fault_time'] = pd.to_datetime(label_df['fault_time'])
submit_df['fault_time'] = pd.to_datetime(submit_df['fault_time'])

log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
label_df['fault_time_ts'] = label_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df['fault_time_ts'] = submit_df["fault_time"].values.astype(np.int64) // 10 ** 9

if stage == 'final_a':
    crashdump_df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_crashdump_dataset.csv'))
    venus_df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_venus_dataset.csv'))
    crashdump_df2 = pd.read_csv('/tcdata/final_crashdump_dataset_a.csv')
    venus_df2 = pd.read_csv('/tcdata/final_venus_dataset_a.csv')
    crashdump_df = pd.concat([crashdump_df1, crashdump_df2]).reset_index(drop=True)
    venus_df = pd.concat([venus_df1, venus_df2]).reset_index(drop=True)
elif stage == 'final_b':
    crashdump_df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_crashdump_dataset.csv'))
    venus_df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_venus_dataset.csv'))
    crashdump_df2 = pd.read_csv('/tcdata/final_crashdump_dataset_b.csv')
    venus_df2 = pd.read_csv('/tcdata/final_venus_dataset_b.csv')
    crashdump_df = pd.concat([crashdump_df1, crashdump_df2]).reset_index(drop=True)
    venus_df = pd.concat([venus_df1, venus_df2]).reset_index(drop=True)
else:
    crashdump_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_crashdump_dataset.csv'))
    venus_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/preliminary_venus_dataset.csv'))
crashdump_df['fault_time'] = pd.to_datetime(crashdump_df['fault_time'])
venus_df['fault_time'] = pd.to_datetime(venus_df['fault_time'])
crashdump_df['fault_time_ts'] = crashdump_df["fault_time"].values.astype(np.int64) // 10 ** 9
venus_df['fault_time_ts'] = venus_df["fault_time"].values.astype(np.int64) // 10 ** 9

label_df = label_df.merge(log_df[['sn', 'server_model']].drop_duplicates(), on=['sn'], how='left')
submit_df = submit_df.merge(log_df[['sn', 'server_model']].drop_duplicates(), on=['sn'], how='left')
label_df = label_df.fillna('MISSING')
submit_df = submit_df.fillna('MISSING')
print(label_df.shape, submit_df.shape)

label_cnt_df = label_df.groupby('label').size().reset_index().rename({0: 'label_cnt'}, axis=1)
label_model_cnt_df = label_df.groupby(['server_model', 'label']).size().reset_index()\
    .rename({0: 'label_model_cnt'}, axis=1)
label_model_cnt_df = label_model_cnt_df.merge(label_cnt_df, on='label', how='left')
label_model_cnt_df['model/label'] = label_model_cnt_df['label_model_cnt'] / label_model_cnt_df['label_cnt']

def safe_split(strs, n, sep='|'):
    str_li = strs.split(sep)
    if len(str_li) >= n + 1:
        return str_li[n].strip()
    else:
        return ''

log_df['time_gap'] = log_df['time'].dt.ceil('30T')
log_df['msg_split_0'] = log_df['msg_lower'].apply(lambda x: safe_split(x, 0))
log_df['msg_split_1'] = log_df['msg_lower'].apply(lambda x: safe_split(x, 1))
log_df['msg_split_2'] = log_df['msg_lower'].apply(lambda x: safe_split(x, 2))

sentences_list = list()
for info, group in log_df.groupby(['sn', 'time_gap']):
    group = group.sort_values(by='time')
    group = group.tail(20)
    sentences_list.append("\n".join(group['msg_lower'].values.astype(str)))

sentences = list()
for s in sentences_list:
    sentences.append([w for w in s.split()])

w2v_model = Word2Vec(sentences, vector_size=64, window=3, min_count=2, sg=0, hs=1, workers=1, seed=2022)

tfv = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_features=50000)
tfv.fit(sentences_list)
X_tfidf = tfv.transform(sentences_list)
svd = TruncatedSVD(n_components=16, random_state=42)
svd.fit(X_tfidf)

def get_w2v_mean(sentences):
    emb_matrix = list()
    vec = list()
    for w in sentences.split():
        if w in w2v_model.wv:
            vec.append(w2v_model.wv[w])
    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * w2v_model.vector_size)
    return emb_matrix

def get_w2v_feature(df):
    vec = get_w2v_mean('\n'.join(df['msg_lower'].values.astype(str)))[0]
    data = {}
    for i in range(64):
        data['w2v_%d'%i] = vec[i]
    return data

def get_feature3(df):
    if df.shape[0] > 0:
        server_model = df['server_model'].values[0]
        sub = label_model_cnt_df[label_model_cnt_df['server_model'] == server_model]
        probas = [0, 0, 0, 0]
        for item in sub.values:
            probas[item[1]] = item[4]
    else:
        probas = [0, 0, 0, 0]
    return {
        'server_model_p0': probas[0],
        'server_model_p1': probas[1],
        'server_model_p2': probas[2],
        'server_model_p3': probas[3],
    }

def get_tfidf_svd(sentence, n_components=16):
    X_tfidf = tfv.transform(sentence)
    X_svd = svd.transform(X_tfidf)
    return X_svd

def get_tfidf_feature(df):
    vec = get_tfidf_svd(['\n'.join(df['msg_lower'].values.astype(str))])[0]
    data = {}
    for i in range(16):
        data['tfv_%d'%i] = vec[i]
    return data

def make_dataset(dataset, data_type='train'):
    ret = []
    for idx in tqdm(range(dataset.shape[0])):
        row = dataset.iloc[idx]
        sn = row['sn']
        fault_time = row['fault_time']
        fault_time_ts = row['fault_time_ts']
        server_model = row['server_model']
        sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
        sub_log = sub_log.sort_values(by='time')
        last_msg_id = str(sub_log['msg_id'].values[-1]) if sub_log.shape[0] > 0 else 'NULL'
        last_template_id = str(sub_log['template_id'].values[-1]) if sub_log.shape[0] > 0 else 'NULL'
        data = {
            'sn': sn,
            'fault_time': fault_time,
            'server_model': server_model,
            'last_msg_id': last_msg_id,
            'last_template_id': last_template_id,
        }
        df_tmp1 = sub_log.tail(20)
        df_tmp2 = sub_log.tail(50)

        data_tmp = get_w2v_feature(df_tmp1)
        data.update(data_tmp)

        data_tmp = get_feature3(df_tmp1)
        data.update(data_tmp)

        data_tmp = get_tfidf_feature(df_tmp1)
        data.update(data_tmp)

        sub_crashdump = crashdump_df[(crashdump_df['sn'] == sn) & \
            (crashdump_df['fault_time_ts'] <= fault_time_ts) & \
            (fault_time_ts - crashdump_df['fault_time_ts'] <= 60 * 60 * 24)]
        sub_crashdump = sub_crashdump.sort_values(by='fault_time_ts')
        data['crashdump_cnt'] = sub_crashdump.shape[0]

        sub_venus = venus_df[(venus_df['sn'] == sn) & \
            (venus_df['fault_time_ts'] <= fault_time_ts) & \
            (fault_time_ts - venus_df['fault_time_ts'] <= 60 * 60 * 24)]
        sub_venus = sub_venus.sort_values(by='fault_time_ts')
        if sub_venus.shape[0] > 0:
            data['venus_module_cnt'] = len(sub_venus.iloc[-1]['module'].split(','))
        else:
            data['venus_module_cnt'] = 0
        # data['venus_cnt'] = sub_venus.shape[0]

        if data_type == 'train':
            data['label'] = row['label']
        ret.append(data)
    return ret

train = make_dataset(label_df, data_type='train')
df_train = pd.DataFrame(train)

test = make_dataset(submit_df, data_type='test')
df_test = pd.DataFrame(test)

df_train.to_csv(os.path.join(os.path.dirname(__file__), '../user_data/train.csv'), index=False)
df_test.to_csv(os.path.join(os.path.dirname(__file__), '../user_data/test.csv'), index=False)
