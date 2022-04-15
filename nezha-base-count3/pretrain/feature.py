import torch
import numpy as np
import pandas as pd
import random
import os
random.seed(0)
np.random.seed(0)#seed应该在main里尽早设置，以防万一
os.environ['PYTHONHASHSEED'] =str(0)#消除hash算法的随机性
import transformers as _
from transformers1 import BertTokenizer
from Config import TOKENIZERS
from tqdm import tqdm

from NEZHA.configuration_nezha import NeZhaConfig
from NEZHA.modeling_nezha import NeZhaForMaskedLM


maxlen=64
batch_size=64
vocab_file_dir = './nezha_model/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.tensor(ls,device='cuda') if returnTensor else ls

config = NeZhaConfig(
    vocab_size=len(tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
)

model = NeZhaForMaskedLM.from_pretrained("./nezha_model").to(torch.device('cuda'))

tokenizer = TOKENIZERS['NEZHA'].from_pretrained(vocab_file_dir)

label1 = pd.read_csv('../../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

submit_df = pd.read_csv('../../data/preliminary_submit_dataset_b.csv')

log_df = pd.read_csv('../../new_src/new_log.csv')

log_df['time'] = pd.to_datetime(log_df['time'])
label_df['fault_time'] = pd.to_datetime(label_df['fault_time'])
submit_df['fault_time'] = pd.to_datetime(submit_df['fault_time'])

log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
label_df['fault_time_ts'] = label_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df['fault_time_ts'] = submit_df["fault_time"].values.astype(np.int64) // 10 ** 9

def get_bert_feature(df):
    features = []
    for i in range(df.shape[0]):
        input_ids, input_masks, segment_ids = [], [], []
        text = df.iloc[i]['msg'].strip()
        tkRes = tokenizer(text, max_length=64, truncation='longest_first',
                               return_attention_mask=False)
        input_id = tkRes['input_ids']
        segment_id = tkRes['token_type_ids']
        input_ids.append(input_id)
        segment_ids.append(segment_id)

        input_ids = paddingList(input_ids, 0, returnTensor=True)
        segment_ids = paddingList(segment_ids, 0, returnTensor=True)
        input_masks = (input_ids != 0)

        features.append(model(input_ids, input_masks, segment_ids)[0].cpu().detach().numpy().mean(axis=1))
    features = np.array(features)
    features = features.reshape(features.shape[0], features.shape[2])
    features = np.mean(features, axis=0)
    return features

tr_data = []
for idx, row in tqdm(label_df.iterrows()):
    sn = row['sn']
    fault_time = row['fault_time']
    ts = row['fault_time_ts']
    label = row['label']

    df = log_df[log_df['sn'] == sn].copy()
    df = df[df['time_ts'] <= ts].copy()
    df = df.sort_values(by='time_ts').reset_index(drop=True)
    df = df.tail(20).copy()

    if df.shape[0] > 0:
        tr_data.append(get_bert_feature(df))
    else:
        tr_data.append(np.zeros(465,))

train_feature = np.array(tr_data)

te_data = []
for idx, row in tqdm(submit_df.iterrows()):
    sn = row['sn']
    fault_time = row['fault_time']
    ts = row['fault_time_ts']

    df = log_df[log_df['sn'] == sn].copy()
    df = df[df['time_ts'] <= ts].copy()
    df = df.sort_values(by='time_ts').reset_index(drop=True)
    df = df.tail(20).copy()

    if df.shape[0] > 0:
        te_data.append(get_bert_feature(df))
    else:
        te_data.append(np.zeros(465,))

test_feature = np.array(te_data)

label_data = []
sn_data = []
fault_time_data = []
for idx, row in tqdm(label_df.iterrows()):
    label = row['label']
    sn = row['sn']
    fault_time = row['fault_time']
    label_data.append(label)
    sn_data.append(sn)
    fault_time_data.append(fault_time)

train_df = pd.DataFrame(train_feature)
train_df.columns = ['bert_%d'%i for i in range(465)]

train_df['sn'] = sn_data
train_df['fault_time'] = fault_time_data
train_df['label'] = label_data

sn_data = []
fault_time_data = []
for idx, row in tqdm(submit_df.iterrows()):
    sn = row['sn']
    fault_time = row['fault_time']
    sn_data.append(sn)
    fault_time_data.append(fault_time)

test_df = pd.DataFrame(test_feature)
test_df.columns = ['bert_%d'%i for i in range(465)]

test_df['sn'] = sn_data
test_df['fault_time'] = fault_time_data

train_df.to_csv('train5.csv', index=False)
test_df.to_csv('test5.csv', index=False)


