from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from model import *
from utils import *
import time
import gc
import logging
logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')


from NEZHA.modeling_nezha import *

MODEL_CLASSES = {
    'BertForClass': BertForClass,
    'BertLastCls': BertLastCls,
    'BertLastTwoCls': BertLastTwoCls,
    'BertLastTwoClsPooler': BertLastTwoClsPooler,
    'BertLastTwoEmbeddings': BertLastTwoEmbeddings,
    'BertLastTwoEmbeddingsPooler': BertLastTwoEmbeddingsPooler,
    'BertLastFourCls': BertLastFourCls,
    'BertLastFourClsPooler': BertLastFourClsPooler,
    'BertLastFourEmbeddings': BertLastFourEmbeddings,
    'BertLastFourEmbeddingsPooler': BertLastFourEmbeddingsPooler,
    'BertDynCls': BertDynCls,
    'BertDynEmbeddings': BertDynEmbeddings,
    'BertRNN': BertRNN,
    'BertCNN': BertCNN,
    'BertRCNN': BertRCNN,
    'XLNet': XLNet,
    'Electra': Electra,
    'NEZHA': NEZHA,

}


class Config:
    def __init__(self):
        # 预训练模型路径
        self.modelId = 2
        self.model = "NEZHA"
        self.Stratification = False
        self.model_path = '../pretrain/nezha_model/'

        self.num_class = 4
        self.dropout = 0.2
        self.MAX_LEN = 64
        self.epoch = 3
        self.learn_rate = 4e-5
        self.normal_lr = 1e-4
        self.batch_size = 32
        self.k_fold = 5
        self.seed = 42

        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')

        self.focalloss = False
        self.pgd = False
        self.fgm = True


config = Config()
os.environ['PYTHONHASHSEED']='0'#消除hash算法的随机性
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

submit_df = pd.read_csv('../../data/preliminary_submit_dataset_a.csv')

test_log_df = pd.read_csv('../../data/preliminary_sel_log_dataset_a.csv')

def text_process(msg):
    msg = [item.strip() for item in msg.lower().strip().split(' ')]
    new_msg = []
    for item in msg:
        new_msg += item.split('_')
    msg = new_msg
    new_msg = []
    for item in msg:
        new_msg += item.split(':')
    msg = new_msg
    new_msg = []
    for item in msg:
        new_msg += item.split('/')
    msg = new_msg
    for kw in ['hdd', 'disk', 'cpu', 'fan', 'dimm']:
        new_msg = []
        for item in msg:
            if item.startswith(kw):
                new_msg += [kw, item[len(kw):]]
            else:
                new_msg += [item]
        msg = new_msg
    msg = list(filter(lambda x: len(x) > 0, msg))
    return ' '.join(msg)

te_data = []
for i, row in submit_df.iterrows():
    sub_df = test_log_df[(test_log_df['sn']==row['sn'])&(test_log_df['time']<=row['fault_time'])].tail(10)
    texts = []
    for item in sub_df['msg'].values:
        text = text_process(item)
        texts.append(text)
    te_data.append([' '.join(texts[::-1]), -1])

test = pd.DataFrame(te_data)
test.columns=['text', 'label']

test_text = test['text'].values.astype(str)
test_label = test['label'].values.astype(int)


config = Config()

y_p = []
for fold in range(5):
    print('\n\n------------fold:{}------------\n'.format(fold))

    model = torch.load('./models/bert_%d.pth'%fold)

    y_p.append([])
    data_gen = data_generator([test_text, test_label], config, shuffle=False)

    with torch.no_grad():
        for input_ids, input_masks, segment_ids, labels in tqdm(data_gen,disable=True):
            y_pred = model(input_ids, input_masks, segment_ids)
            y_pred = F.softmax(y_pred)
            y_pred = y_pred.detach().to("cpu").numpy()
            y_p[-1] += list(y_pred[:,])
    y_p[-1] = np.array(y_p[-1])
y_p = y_p[0] + y_p[1] + y_p[2] + y_p[3] + y_p[4]
y_p = np.argmax(y_p, axis=1)
submit_df['label'] = y_p

submit_df.to_csv('result.csv', index=False)
