from cmath import log
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
from sklearn.decomposition import TruncatedSVD
from tsfresh.feature_extraction.feature_calculators import abs_energy, benford_correlation, count_above, \
    count_above_mean, mean_abs_change, mean_change, percentage_of_reoccurring_datapoints_to_all_datapoints, \
    percentage_of_reoccurring_values_to_all_values, sample_entropy

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

def get_ts_feature(df, fault_time_ts):
    data = {}
    log_times = df['time_ts'].values
    span_times = []
    for i in range(len(log_times)):
        span_times.append(fault_time_ts - log_times[i])
    data['abs_energy'] = abs_energy(span_times)
    data['benford_correlation'] = benford_correlation(span_times)
    data['count_above'] = count_above(span_times, [60*60])
    data['count_above_mean'] = count_above_mean(span_times)
    data['mean_abs_change'] = mean_abs_change(span_times)
    data['mean_change'] = mean_change(span_times)
    data['percentage_of_reoccurring_datapoints_to_all_datapoints'] = percentage_of_reoccurring_datapoints_to_all_datapoints(span_times)
    data['percentage_of_reoccurring_values_to_all_values'] = percentage_of_reoccurring_values_to_all_values(span_times)
    data['sample_entropy'] = sample_entropy(span_times)
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
            # 'fault_hour': fault_time.hour,
            # 'fault_dayofweek': fault_time.dayofweek,
        }
        # df_tmp1 = sub_log[sub_log['time_ts'] - fault_time_ts >= -60 * 60 * 2]
        df_tmp1 = sub_log.tail(20)

        data_tmp = get_ts_feature(df_tmp1, fault_time_ts)
        data.update(data_tmp)

        if data_type == 'train':
            data['label'] = row['label']
        ret.append(data)
    return ret

train = make_dataset(label_df, data_type='train')
df_train = pd.DataFrame(train)

test = make_dataset(submit_df, data_type='test')
df_test = pd.DataFrame(test)

df_train.to_csv(os.path.join(os.path.dirname(__file__), './train.csv'), index=False)
df_test.to_csv(os.path.join(os.path.dirname(__file__), './test.csv'), index=False)
