import json
import numpy as np
import pandas as pd
from bert4keras.snippets import truncate_sequences
from bert4keras.tokenizers import load_vocab
from tqdm import tqdm

min_count = 2
maxlen = 64
dict_path = '../user_data/NEZHA-Base/vocab.txt'

label1 = pd.read_csv('../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

submit_df = pd.read_csv('../data/preliminary_submit_dataset_a.csv')
submit_df_b = pd.read_csv('../data/preliminary_submit_dataset_b.csv')

log_df = pd.read_csv('../new_src/new_log.csv')

log_df['time'] = pd.to_datetime(log_df['time'])
label_df['fault_time'] = pd.to_datetime(label_df['fault_time'])
submit_df['fault_time'] = pd.to_datetime(submit_df['fault_time'])
submit_df_b['fault_time'] = pd.to_datetime(submit_df_b['fault_time'])

log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
label_df['fault_time_ts'] = label_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df['fault_time_ts'] = submit_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df_b['fault_time_ts'] = submit_df_b["fault_time"].values.astype(np.int64) // 10 ** 9

tr_data = []
for idx, row in tqdm(label_df.iterrows()):
    sn = row['sn']
    fault_time = row['fault_time']
    ts = row['fault_time_ts']
    label = row['label']
    df = log_df[log_df['sn'] == sn].copy()
    df = df[df['time_ts'] <= ts].copy()
    df = df.sort_values(by='time_ts').reset_index(drop=True)
    df = df.tail(50).copy()
    if df.shape[0] > 0:
        tr_data.append([sn, ' '.join(df['msg_id'].values.astype(str)), label])
    else:
        tr_data.append([sn, ' '.join(['-1']), label])

te_data = []
for idx, row in tqdm(submit_df.iterrows()):
    sn = row['sn']
    fault_time = row['fault_time']
    ts = row['fault_time_ts']

    df = log_df[log_df['sn'] == sn].copy()
    df = df[df['time_ts'] <= ts].copy()
    df = df.sort_values(by='time_ts').reset_index(drop=True)
    df = df.tail(50).copy()
    if df.shape[0] > 0:
        te_data.append([sn, ' '.join(df['msg_id'].values.astype(str))])
    else:
        te_data.append([sn, ' '.join(['-1'])])

te_data_b = []
for idx, row in tqdm(submit_df_b.iterrows()):
    sn = row['sn']
    fault_time = row['fault_time']
    ts = row['fault_time_ts']

    df = log_df[log_df['sn'] == sn].copy()
    df = df[df['time_ts'] <= ts].copy()
    df = df.sort_values(by='time_ts').reset_index(drop=True)
    df = df.tail(50).copy()
    if df.shape[0] > 0:
        te_data_b.append([sn, ' '.join(df['msg_id'].values.astype(str))])
    else:
        te_data_b.append([sn, ' '.join(['-1'])])


def load_data(data_list):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    for l in data_list:
        if len(l) == 3:
            a, b = l[1], int(l[2])
        else:
            a, b = l[1], -5  # 未标注数据，标签为-5
        a = [int(i) for i in a.split(' ')]
        truncate_sequences(maxlen, -1, a)
        D.append((a, b))
    return D

data = load_data(tr_data)
test_data = load_data(te_data)
test_data_b = load_data(te_data_b)

tokens = {}
for d in data + test_data + test_data_b:
    for i in d[0]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= min_count}
tokens = sorted(tokens.items(), key=lambda s: -s[1])
tokens = {
    t[0]: i + 9
    for i, t in enumerate(tokens)
}

counts = json.load(open('../user_data/counts.json'))
del counts['[CLS]']
del counts['[SEP]']
token_dict = load_vocab(dict_path)
freqs = [
    counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
]
keep_tokens = list(np.argsort(freqs)[::-1])
