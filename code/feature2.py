import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

label1 = pd.read_csv('../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

submit_df = pd.read_csv('../data/preliminary_submit_dataset_b.csv')

log_df = pd.read_csv('./log_template.csv')

log_df['time'] = pd.to_datetime(log_df['time'])
label_df['fault_time'] = pd.to_datetime(label_df['fault_time'])
submit_df['fault_time'] = pd.to_datetime(submit_df['fault_time'])

log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
label_df['fault_time_ts'] = label_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df['fault_time_ts'] = submit_df["fault_time"].values.astype(np.int64) // 10 ** 9

label_df = label_df.merge(log_df[['sn', 'server_model']].drop_duplicates(), on=['sn'], how='left')
submit_df = submit_df.merge(log_df[['sn', 'server_model']].drop_duplicates(), on=['sn'], how='left')

sentences = list()
for s in log_df['msg_lower'].values:
    sentences.append([w for w in s.split()])

w2v_model = Word2Vec(sentences, vector_size=64, window=3, min_count=2, sg=0, hs=1, workers=1, seed=2022)

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

train_log_dfs = []
for idx in tqdm(range(label_df.shape[0])):
    row = label_df.iloc[idx]
    sn = row['sn']
    fault_time = row['fault_time']
    fault_time_ts = row['fault_time_ts']
    sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
    sub_log = sub_log.sort_values(by='time')
    df_tmp1 = sub_log[sub_log['time_ts'] - fault_time_ts >= -60 * 60 * 2]
    train_log_dfs.append(df_tmp1.iloc[:, 1:])
label_df['log_df'] = train_log_dfs

test_log_dfs = []
for idx in tqdm(range(submit_df.shape[0])):
    row = submit_df.iloc[idx]
    sn = row['sn']
    fault_time = row['fault_time']
    fault_time_ts = row['fault_time_ts']
    sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
    sub_log = sub_log.sort_values(by='time')
    df_tmp1 = sub_log[sub_log['time_ts'] - fault_time_ts >= -60 * 60 * 2]
    test_log_dfs.append(df_tmp1.iloc[:, 1:])
submit_df['log_df'] = test_log_dfs

info_vec_map = {}
for info, group in label_df.groupby('server_model'):
    vecs = []
    for item in group['log_df']:
        vecs.append(get_w2v_mean('\n'.join(item['msg_lower'].values))[0])
    info_vec_map['%s'%(info)] = np.mean(vecs, axis=0)

counters = [Counter(), Counter(), Counter(), Counter()]
for idx in tqdm(range(label_df.shape[0])):
    row = label_df.iloc[idx]
    sn = row['sn']
    fault_time = row['fault_time']
    fault_time_ts = row['fault_time_ts']
    label = row['label']
    sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
    sub_log = sub_log.sort_values(by='time')
    df_tmp = sub_log[sub_log['time_ts'] - fault_time_ts >= -60 * 60 * 2]
    counters[label].update(df_tmp['msg_lower'].values)

vecs = []
for i in range(len(counters[0])):
    vec = get_w2v_mean(list(counters[0].keys())[i])
    weight = (list(counters[0].values())[i] / np.sum(list(counters[0].values())))
    vec = np.array(vec) * weight
    vecs.append(vec[0])
vec0 = np.sum(np.array(vecs), axis=0)

vecs = []
for i in range(len(counters[1])):
    vec = get_w2v_mean(list(counters[1].keys())[i])
    weight = (list(counters[1].values())[i] / np.sum(list(counters[1].values())))
    vec = np.array(vec) * weight
    vecs.append(vec[0])
vec1 = np.sum(np.array(vecs), axis=0)

vecs = []
for i in range(len(counters[2])):
    vec = get_w2v_mean(list(counters[2].keys())[i])
    weight = (list(counters[2].values())[i] / np.sum(list(counters[2].values())))
    vec = np.array(vec) * weight
    vecs.append(vec[0])
vec2 = np.sum(np.array(vecs), axis=0)

vecs = []
for i in range(len(counters[3])):
    vec = get_w2v_mean(list(counters[3].keys())[i])
    weight = (list(counters[3].values())[i] / np.sum(list(counters[3].values())))
    vec = np.array(vec) * weight
    vecs.append(vec[0])
vec3 = np.sum(np.array(vecs), axis=0)


def make_dataset(dataset, data_type='train'):
    ret = []
    for idx in tqdm(range(dataset.shape[0])):
        row = dataset.iloc[idx]
        sn = row['sn']
        fault_time = row['fault_time']
        fault_time_ts = row['fault_time_ts']
        sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
        sub_log = sub_log.sort_values(by='time')

        df_tmp1 = sub_log[sub_log['time_ts'] - fault_time_ts >= -60 * 60 * 2]
        data = {
            'sn': sn,
            'fault_time': fault_time,
        }
        vec = get_w2v_mean('\n'.join(df_tmp1['msg_lower'].values.astype(str)))[0]
        data['cos_sim_label0'] = cosine_similarity([vec], [vec0])[0][0]
        data['cos_sim_label1'] = cosine_similarity([vec], [vec1])[0][0]
        data['cos_sim_label2'] = cosine_similarity([vec], [vec2])[0][0]
        data['cos_sim_label3'] = cosine_similarity([vec], [vec3])[0][0]
        # data['dist_label0'] = np.linalg.norm(vec - vec0)
        # data['dist_label1'] = np.linalg.norm(vec - vec1)
        # data['dist_label2'] = np.linalg.norm(vec - vec2)
        # data['dist_label3'] = np.linalg.norm(vec - vec3)
        # data['p_label0'] =  np.corrcoef(vec, vec0)[0][1]
        # data['p_label1'] =  np.corrcoef(vec, vec1)[0][1]
        # data['p_label2'] =  np.corrcoef(vec, vec2)[0][1]
        # data['p_label3'] =  np.corrcoef(vec, vec3)[0][1]
        data['cos_sim_server_model'] = cosine_similarity([vec], [info_vec_map[row['server_model']]])[0][0]

        if data_type == 'train':
            data['label'] = row['label']
        ret.append(data)
    return ret

train = make_dataset(label_df, data_type='train')
df_train = pd.DataFrame(train)

test = make_dataset(submit_df, data_type='test')
df_test = pd.DataFrame(test)

df_train.to_csv('train2.csv', index=False)
df_test.to_csv('test2.csv', index=False)
