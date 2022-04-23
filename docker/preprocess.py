import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from drain3 import TemplateMiner #开源在线日志解析框架
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig


label1 = pd.read_csv('./data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('./data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)

train_log_df = pd.read_csv('./data/preliminary_sel_log_dataset.csv')
test_log_df = pd.read_csv('./data/preliminary_sel_log_dataset_a.csv')
test_b_log_df = pd.read_csv('./data/preliminary_sel_log_dataset_b.csv')
final_test_a_log_df = pd.read_csv('/tcdata/final_sel_log_dataset_a.csv')
log_df = pd.concat([train_log_df, test_log_df, test_b_log_df, final_test_a_log_df]).reset_index(drop=True)
log_df = log_df.dropna(subset=['server_model'])
log_df = log_df.fillna('MISSING')

data = []
for i in tqdm(range(log_df.shape[0])):
    row = log_df.iloc[i]
    msg = row['msg'].strip().lower()
    if msg.find('oem record') > 0:
        msg = 'hex | oem record hex | seq'
    if msg.startswith('oem record'):
        msg = 'oem record | %s | seq'%(msg.split(' | ')[1])
    if msg.startswith('temperature') and len(msg.split(' | ')) == 4:
        msg = ' | '.join(msg.split(' | ')[:3])
    if msg.startswith('temperature') and msg.split(' | ')[2].startswith('reading'):
        msg = ' | '.join(msg.split(' | ')[:2]) + ' | seq'
    if msg.startswith('memory dimm'):
        msg = 'memory dimm | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('fan') and len(msg.split(' | ')) == 4:
        msg = 'fan | ' + ' | '.join(msg.split(' | ')[1:3])
    if msg.startswith('fan') and  msg.split(' | ')[2].startswith('reading'):
        msg = 'fan | %s | seq'%(msg.split(' | ')[1])
    if msg.startswith('memory #0x'):
        msg = 'memory hex | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot hdd'):
        msg = 'drive slot hdd | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot disk'):
        msg = 'drive slot disk | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay hdd'):
        msg = 'drive slot / bay hdd | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay disk'):
        msg = 'drive slot / bay disk | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay nvmessd'):
        msg = 'drive slot / bay nvmessd | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay bp'):
        msg = 'drive slot / bay bp | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay rear'):
        msg = 'drive slot / bay rear | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay front'):
        msg = 'drive slot / bay front | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot / bay #0x'):
        msg = 'drive slot / bay hex | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('power supply ps'):
        msg = 'power supply ps | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot front'):
        msg = 'drive slot front | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('processor #0x'):
        msg = 'processor hex | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('memory cpu0_'):
        msg = 'memory cpu0_hex | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('memory cpu1_'):
        msg = 'memory cpu1_hex | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('drive slot #0'):
        msg = 'drive slot hex | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('memory mem_ch'):
        msg = 'memory mem_ch | ' + ' | '.join(msg.split(' | ')[1:])
    if msg.startswith('power supply #0'):
        msg = 'power supply hex | ' + ' | '.join(msg.split(' | ')[1:])
    msg = msg.replace('_', ' ')
    data.append(msg)

log_df['msg_lower'] = data

msg_unique = log_df['msg_lower'].unique()
msg_map = {}
for i in range(len(msg_unique)):
    msg_map[msg_unique[i]] = i

log_df['msg_id'] = log_df['msg_lower'].apply(lambda x: msg_map[x])

config = TemplateMinerConfig()
config.load('./drain3.ini')
config.profiling_enabled = False

drain_file = 'comp_a_sellog'
persistence = FilePersistence(drain_file + '.bin')
template_miner = TemplateMiner(persistence, config=config)

for msg in log_df.msg_lower.tolist():
    template_miner.add_log_message(msg)
temp_count = len(template_miner.drain.clusters)

template_dic = {}
for cluster in template_miner.drain.clusters:
    template_dic[cluster.cluster_id] = cluster.size

def match_template(df, template_miner, template_dic):
    msg = df.msg_lower
    cluster = template_miner.match(msg) # 匹配模板，由开源工具提供
    if cluster and cluster.cluster_id in template_dic:
        df['template_id'] = cluster.cluster_id # 模板id
        df['template'] = cluster.get_template() # 具体模板
    else:
        df['template_id'] = 'None' # 没有匹配到模板的数据也会记录下来，之后也会用作一种特征。
        df['template'] = 'None'
    return df

log_df = log_df.apply(match_template, template_miner=template_miner, template_dic=template_dic, axis=1)

log_df.to_csv('log_template.csv', index=False)
