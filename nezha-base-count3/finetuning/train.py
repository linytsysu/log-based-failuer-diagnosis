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
        self.epoch = 5
        self.learn_rate = 1e-5
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

train_log_df = pd.read_csv('../../data/preliminary_sel_log_dataset.csv')
test_log_df = pd.read_csv('../../data/preliminary_sel_log_dataset_a.csv')

label1 = pd.read_csv('../../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)

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

tr_data = []
for sn, group in tqdm(train_log_df.groupby('sn')):
    group = group.sort_values(by='time')
    sub_label_df = label_df[label_df['sn'] == sn]
    for item in sub_label_df.values:
        fault_time = item[1]
        label = item[2]
        sub_df = group[group['time'] <= fault_time].tail(10)
        texts = []
        for i in range(sub_df.shape[0]):
            text = text_process(sub_df.iloc[i]['msg'])
            texts.append(text)
        tr_data.append([' '.join(texts[::-1]), label])

train = pd.DataFrame(tr_data)
train.columns=['text', 'label']

train_text = train['text'].values.astype(str)
train_label = train['label'].values.astype(int)

oof_train = np.zeros((len(train), config.num_class), dtype=np.float32)

kf = KFold(n_splits=config.k_fold, shuffle=True, random_state=config.seed)

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

for fold, (train_index, valid_index) in enumerate(kf.split(train_text, train_label)):
    print('\n\n------------fold:{}------------\n'.format(fold))

    text = train_text[train_index]
    y = train_label[train_index]

    val_text = train_text[valid_index]
    val_y = train_label[valid_index]

    train_D = data_generator([text, y], config, shuffle=True)
    val_D = data_generator([val_text, val_y], config)

    model = MODEL_CLASSES[config.model](config).to(config.device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)


    if config.pgd:
        pgd = PGD(model)
        K = 3
    elif config.fgm:
        fgm = FGM(model)

    if config.focalloss:
        loss_fn = FocalLoss(config.num_class)
    else:
        loss_fn = nn.CrossEntropyLoss()  # BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步


    num_train_steps = int(len(train) / config.batch_size * config.epoch)
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if config.Stratification:
        bert_params = [x for x in param_optimizer if 'bert' in x[0]]
        normal_params = [p for n, p in param_optimizer if 'bert' not in n]
        optimizer_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': normal_params, 'lr': config.normal_lr},
        ]
    else:
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

    optimizer = AdamW(optimizer_parameters, lr=config.learn_rate) # lr为全局学习率
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train) / config.batch_size / 2),
        num_training_steps=num_train_steps
    )

    best_f1 = 0
    PATH = './models/bert_{}.pth'.format(fold)
    save_model_path = './models/'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    for e in range(config.epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D,ncols=70,disable=True)
        last=time.time()
        for input_ids, input_masks, segment_ids, labels in tq:
            label_t = torch.tensor(labels, dtype=torch.long).to(config.device)

            y_pred = model(input_ids, input_masks, segment_ids)

            loss = loss_fn(y_pred, label_t)
            loss = loss.mean()
            loss.backward()

            if config.pgd:
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    y_pred = model(input_ids, input_masks, segment_ids)

                    loss_adv = loss_fn(y_pred, label_t)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore()  # 恢复embedding参数

            elif config.fgm:
                # 对抗训练
                fgm.attack()  # 在embedding上添加对抗扰动
                y_pred = model(input_ids, input_masks, segment_ids)
                loss_adv = loss_fn(y_pred, label_t)
                loss_adv = loss_adv.mean()
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                fgm.restore()  # 恢复embedding参数


            # 梯度下降，更新参数
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            y_pred = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            acc += sum(y_pred == labels)
            loss_num += loss.item()
            train_len += len(labels)
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)
        print(f"微调第{e}轮耗时：{time.time()-last}")
        model.eval()
        with torch.no_grad():
            y_p = []
            y_l = []
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D,disable=True):
                label_t = torch.tensor(labels, dtype=torch.long).to(config.device)

                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = F.softmax(y_pred)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))

                y_p += list(y_pred[:,1])

                y_pred = np.argmax(y_pred, axis=1)
                y_l += list(y_pred)


            f1 = macro_f1(val_y, y_l)
            print("best_f1:{} \n".format(f1))
            if f1 >= best_f1:
                best_f1 = f1
                oof_train[valid_index] = np.array(train_logit)
                #torch.save(model.module.state_dict() if hasattr(model, "module") else model.state_dict(), PATH)
                torch.save(model.module if hasattr(model, "module") else model, PATH)

    optimizer.zero_grad()

    del text
    del y
    del val_text
    del val_y
    del train_D
    del val_D
    del tq
    del model
    torch.cuda.empty_cache()
    gc.collect()