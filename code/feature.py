import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

label1 = pd.read_csv('../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

submit_df = pd.read_csv('../data/preliminary_submit_dataset_b.csv')

print(label_df.shape, submit_df.shape)

log_df = pd.read_csv('./log_template.csv')
log_df['msg_lower'] = log_df['msg_lower'].astype(str)
log_df['server_model'] = log_df['server_model'].astype(str)

log_df['time'] = pd.to_datetime(log_df['time'])
label_df['fault_time'] = pd.to_datetime(label_df['fault_time'])
submit_df['fault_time'] = pd.to_datetime(submit_df['fault_time'])

log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
label_df['fault_time_ts'] = label_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df['fault_time_ts'] = submit_df["fault_time"].values.astype(np.int64) // 10 ** 9

crashdump_df = pd.read_csv('../data/preliminary_crashdump_dataset.csv')
venus_df = pd.read_csv('../data/preliminary_venus_dataset.csv')
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

# counter_map = {}
# for idx in tqdm(range(label_df.shape[0])):
#     row = label_df.iloc[idx]
#     sn = row['sn']
#     fault_time_ts = row['fault_time_ts']
#     sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
#     sub_log = sub_log.sort_values(by='time')
#     df_tmp = sub_log.tail(20)
#     k = '%d'%(row['label'])
#     if not k in counter_map:
#         counter_map[k] = Counter()
#     counter_map[k].update(np.unique(df_tmp['msg_id'].values.tolist()))
# for k in counter_map:
#     counter_map[k] = [item[0] for item in counter_map[k].most_common()[:50]]

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
    sentences_list.append("\n".join(group['msg_lower'].values.astype(str)))

sentences = list()
for s in sentences_list:
    sentences.append([w for w in s.split()])

w2v_model = Word2Vec(sentences, vector_size=64, window=3, min_count=2, sg=0, hs=1, workers=1, seed=2022)

# tokenized_sent = [word_tokenize(s.lower()) for s in sentences_list]
# tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
# d2v_model = Doc2Vec(tagged_data, vector_size=32, window=3, min_count=2, workers=1, seed=2022)
# d2v_model.random.seed(2022)

tfv = TfidfVectorizer(ngram_range=(1,3), min_df=5, max_features=50000)
tfv.fit(sentences_list)
X_tfidf = tfv.transform(sentences_list)
svd = TruncatedSVD(n_components=16, random_state=42)
svd.fit(X_tfidf)

def get_statistical_features(df1, df2, suffix):
    cnt1 = df1.shape[0]
    cnt2 = df2.shape[0]
    percent = cnt1 / cnt2 if cnt2 > 0 else 0

    msg_nunique1 = df1['msg_id'].nunique()
    template_unique1 = df1['template_id'].nunique()
    msg_nunique2 = df2['msg_id'].nunique()
    template_unique2 = df2['template_id'].nunique()

    msg_precent1 = msg_nunique1 / cnt1 if cnt1 > 0 else 0
    template_precent1 = template_unique1 / cnt1 if cnt1 > 0 else 0

    template_to_msg_percent = template_unique1 / msg_nunique1  if msg_nunique1 > 0 else 0
    msg_to_msg_percent = msg_nunique1 / msg_nunique2 if msg_nunique2 > 0 else 0
    template_to_template_percent = template_unique1 / template_unique2  if template_unique2 > 0 else 0

    processor_nunique = df1[~df1['msg_lower'].str.startswith('processor')]['msg_id'].nunique()

    msg_split_0_nunique = df1['msg_split_0'].nunique()
    msg_split_1_nunique = df1['msg_split_1'].nunique()
    msg_split_2_nunique = df1['msg_split_2'].nunique()

    msg_split_0_percent = df1['msg_split_0'].nunique() / cnt1 if cnt1 > 0 else 0
    msg_split_1_percent = df1['msg_split_1'].nunique() / cnt1 if cnt1 > 0 else 0
    msg_split_2_percent = df1['msg_split_2'].nunique() / cnt1 if cnt1 > 0 else 0

    return {
        'cnt_%s'%suffix: cnt1,
        'percent_%s'%suffix: percent,
        'msg_nunique_%s'%suffix: msg_nunique1,
        'template_nunique_%s'%suffix: template_unique1,
        'msg_percent_%s'%suffix: msg_precent1,
        'template_percent_%s'%suffix: template_precent1,
        'template_to_msg_percent_%s'%suffix: template_to_msg_percent,
        'msg_to_msg_percent_%s'%suffix: msg_to_msg_percent,
        'template_to_template_percent_%s'%suffix: template_to_template_percent,
        'processor_nunique_%s'%suffix: processor_nunique,
        'msg_split_0_nunique_%s'%suffix: msg_split_0_nunique,
        'msg_split_1_nunique_%s'%suffix: msg_split_1_nunique,
        'msg_split_2_nunique_%s'%suffix: msg_split_2_nunique,
        'msg_split_0_percent_%s'%suffix: msg_split_0_percent,
        'msg_split_1_percent_%s'%suffix: msg_split_1_percent,
        'msg_split_2_percent_%s'%suffix: msg_split_2_percent,
    }

def get_time_feature(df, fault_time_ts, suffix):
    if df.shape[0] == 0:
        first_time_span = np.nan
        second_span_0 = np.nan
        second_span_5 = np.nan
        second_span_10 = np.nan
        second_span_mean = np.nan
        time_diffs_avg = np.nan
        time_diffs_max = np.nan
        time_diffs_min = np.nan
        time_diffs_std = np.nan
        max_time_diff = np.nan
        max_time_diff_div_first_time_span = np.nan
    else:
        ts_diff_list = []
        for i in range(df.shape[0]):
            ts = df.iloc[i]['time_ts']
            ts_diff_list.append(fault_time_ts - ts)
        first_time_span = ts_diff_list[0]
        second_span_0 = ts_diff_list[-1]
        second_span_5 = ts_diff_list[-5] if len(ts_diff_list) >= 5 else np.nan
        second_span_10 = ts_diff_list[-10] if len(ts_diff_list) >= 10 else np.nan
        second_span_mean = np.mean(ts_diff_list)
        time_diffs = df['time_ts'].diff().iloc[1:]
        time_diffs_avg = np.mean(time_diffs) if time_diffs.shape[0] > 0 else np.nan
        time_diffs_max = np.max(time_diffs) if time_diffs.shape[0] > 0 else np.nan
        time_diffs_min = np.min(time_diffs) if time_diffs.shape[0] > 0 else np.nan
        time_diffs_std = np.std(time_diffs) if time_diffs.shape[0] > 0 else np.nan
        max_time_diff = df['time_ts'].iloc[-1] - df['time_ts'].iloc[0]
        max_time_diff_div_first_time_span = max_time_diff / first_time_span if first_time_span > 0 else np.nan
    return {
        'first_time_span_%s'%suffix: first_time_span,
        'second_span0_%s'%suffix: second_span_0,
        'second_span5_%s'%suffix: second_span_5,
        'second_span10_%s'%suffix: second_span_10,
        'second_span_mean_%s'%suffix: second_span_mean,
        'time_diffs_avg_%s'%suffix: time_diffs_avg,
        'time_diffs_max_%s'%suffix: time_diffs_max,
        'time_diffs_min_%s'%suffix: time_diffs_min,
        'time_diffs_std_%s'%suffix: time_diffs_std,
        'max_time_diff_%s'%suffix: max_time_diff,
        'max_time_diff_div_first_time_span_%s'%suffix: max_time_diff_div_first_time_span,
    }

def get_feature2(df):
    template_counter = Counter()
    template_counter.update(df['template_id'].values)
    template_counter = [(l,k) for k,l in sorted([(j,i) for i,j in template_counter.items()], reverse=True)]
    msg_counter = Counter()
    msg_counter.update(df['msg_id'].values)
    msg_counter = [(l,k) for k,l in sorted([(j,i) for i,j in msg_counter.items()], reverse=True)]

    max_time = 0
    cur_time = 0
    cur_msg = 'NULL'
    for msg in df['msg_id'].values:
        if msg == cur_msg:
            cur_time += 1
            max_time = max((cur_time, max_time))
        else:
            cur_msg = msg
            cur_time = 1
    return {
        'tmp_appearance_1': str(template_counter[0][0]) if len(template_counter) > 0 else 'NULL',
        'tmp_appearance_1_percent': template_counter[0][1] / df.shape[0] if len(template_counter) > 0 else 0,
        'tmp_appearance_2': str(template_counter[1][0]) if len(template_counter) > 1 else 'NULL',
        'tmp_appearance_2_percent': template_counter[1][1] / df.shape[0] if len(template_counter) > 1 else 0,
        'tmp_appearance_3': str(template_counter[2][0]) if len(template_counter) > 2 else 'NULL',
        'tmp_appearance_3_percent': template_counter[2][1] / df.shape[0] if len(template_counter) > 2 else 0,
        'msg_appearance_1': str(msg_counter[0][0]) if len(msg_counter) > 0 else 'NULL',
        'msg_appearance_1_percent': msg_counter[0][1] / df.shape[0] if len(msg_counter) > 0 else 0,
        'msg_appearance_2': str(msg_counter[1][0]) if len(msg_counter) > 1 else 'NULL',
        'msg_appearance_2_percent': msg_counter[1][1] / df.shape[0] if len(msg_counter) > 1 else 0,
        'msg_appearance_3': str(msg_counter[2][0]) if len(msg_counter) > 2 else 'NULL',
        'msg_appearance_3_percent': msg_counter[2][1] / df.shape[0] if len(msg_counter) > 2 else 0,
        'max_continuous_msg': cur_msg,
        'max_continuous_time': max_time
    }

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

# def get_d2v_feature(df):
#     vec = d2v_model.infer_vector(['\n'.join(df['msg_lower'].values.astype(str))])
#     data = {}
#     for i in range(32):
#         data['d2v_%d'%i] = vec[i]
#     return data

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

def get_feature4(df):
    if df.shape[0] == 0:
        last_template_cnt = np.nan
        last_template_percent = np.nan
    else:
        last_template_id = df['template_id'].values[-1]
        last_template_cnt = df[df['template_id'] == last_template_id].shape[0]
        last_template_percent = last_template_cnt / df.shape[0]
    return {
        'last_template_cnt': last_template_cnt,
        'last_template_percent': last_template_percent,
    }

def get_feature5(df):
    keywords = [
        'button', 'critical', 'drive', 'event', 'fan', 'management',
        'memory', 'microcontroller', 'microcontroller/coprocessor', 'oem',
        'os', 'power', 'processor', 'slot', 'slot/connector', 'system',
        'temperature', 'unknown', 'watchdog', 'watchdog2']
    data = {}
    for keyword in keywords:
        sub_df = df[df['msg_lower'].str.startswith(keyword)]
        data['%s_cnt'%keyword] = sub_df.shape[0]
        data['%s_percent'%keyword] = data['%s_cnt'%keyword] / df.shape[0] if df.shape[0] > 0 else 0
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
            'last_template_id': last_template_id
        }
        # df_tmp1 = sub_log[sub_log['time_ts'] - fault_time_ts >= -60 * 60 * 2]
        df_tmp1 = sub_log.tail(20)
        df_tmp2 = sub_log.tail(50)
        data_tmp = get_statistical_features(df_tmp1, df_tmp2, 'tail20')
        data.update(data_tmp)

        data_tmp = get_time_feature(df_tmp1, fault_time_ts, 'tail20')
        data.update(data_tmp)

        data_tmp = get_feature2(df_tmp1)
        data.update(data_tmp)

        data_tmp = get_w2v_feature(df_tmp1)
        data.update(data_tmp)

        # data_tmp = get_d2v_feature(df_tmp1)
        # data.update(data_tmp)

        data_tmp = get_feature3(df_tmp1)
        data.update(data_tmp)

        data_tmp = get_tfidf_feature(df_tmp1)
        data.update(data_tmp)

        data_tmp = get_feature4(df_tmp1)
        data.update(data_tmp)

        # data_tmp = get_feature5(df_tmp1)
        # data.update(data_tmp)

        # for label in range(4):
        #     k = '%d'%(label)
        #     intersect = np.intersect1d(np.unique(df_tmp1['msg_id'].values), counter_map[k])
        #     data['intersect1d_%d'%label] = len(intersect) / len(counter_map[k])

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
        data['venus_cnt'] = sub_venus.shape[0]

        if data_type == 'train':
            data['label'] = row['label']
        ret.append(data)
    return ret

train = make_dataset(label_df, data_type='train')
df_train = pd.DataFrame(train)

test = make_dataset(submit_df, data_type='test')
df_test = pd.DataFrame(test)

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)
