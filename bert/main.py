import os
os.environ['TF_KERAS'] = '1'

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from bert4keras.backend import keras
from bert4keras.optimizers import Adam
from bert4keras.models import build_transformer_model
from tqdm import tqdm

from dataset import data, test_data, test_data_b, tokens, keep_tokens
from model import config_path, checkpoint_path, masked_crossentropy, adversarial_training
from data_generator import data_generator

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode")
args = parser.parse_args()
# mode = args.mode
mode = 'test'

batch_size = 64
n_splits = 5
random_state = 42


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


if mode == 'train':
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    flod = 0
    for train_index, test_index in kf.split(data):
        train_data = [d for i, d in enumerate(data) if i in train_index]
        valid_data = [d for i, d in enumerate(data) if i in test_index]

        # 模拟未标注
        for d in valid_data + test_data + test_data_b:
            train_data.append((d[0], -5))

        # 转换数据集
        train_generator = data_generator(train_data, batch_size)
        valid_generator = data_generator(valid_data, batch_size)

        model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='nezha',
            with_mlm=True,
            keep_tokens=[0, 100, 101, 102, 103, 100, 100, 100, 100] + keep_tokens[:len(tokens)]
        )
        model.compile(loss=masked_crossentropy, optimizer=Adam(3e-5))
        model.summary()
        # adversarial_training(model, 'Embedding-Token', 0.5)

        def evaluate(data):
            """线下评测函数
            """
            Y_true, Y_pred = [], []
            for x_true, y_true in data:
                y_pred = model.predict(x_true)[:, 0, 5:9]
                y_pred = np.argmax(y_pred, axis=1)
                y_true = y_true[:, 0] - 5
                Y_pred.extend(y_pred)
                Y_true.extend(y_true)
            return macro_f1(Y_true, Y_pred)

        class Evaluator(keras.callbacks.Callback):
            """评估与保存
            """
            def __init__(self):
                self.best_val_score = 0.

            def on_epoch_end(self, epoch, logs=None):
                val_score = evaluate(valid_generator)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    model.save_weights('../user_data/best_model_reverse_flod%d.h5'%flod)
                print(
                    u'val_score: %.5f, best_val_score: %.5f\n' %
                    (val_score, self.best_val_score)
                )

        evaluator = Evaluator()
        model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=20,
            callbacks=[evaluator]
        )

        flod += 1
        del model

elif mode == 'test':
    test_generator = data_generator(test_data, batch_size)

    for i in range(n_splits):
        model = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='nezha',
            with_mlm=True,
            keep_tokens=[0, 100, 101, 102, 103, 100, 100, 100, 100] + keep_tokens[:len(tokens)]
        )
        model.compile(loss=masked_crossentropy, optimizer=Adam(3e-5))
        # model.summary()
        # adversarial_training(model, 'Embedding-Token', 0.5)
        model.load_weights('../user_data/best_model_reverse_flod%d.h5'%i)

        out_file = '../prediction_result/result_reverse%d.csv'%i
        preds = []
        for x_true, _ in tqdm(test_generator):
            y_pred = model.predict(x_true)[:, 0, 5:9]
            for pred in y_pred:
                pred = pred / (pred.sum() + 1e-8)
                preds.append(pred)
        pd.DataFrame(preds).to_csv('result_reverse%d.csv'%i, index=False)
        del model

    # values = None
    # for i in range(n_splits):
    #     df = pd.read_csv('./result_reverse%d.csv'%i, header=None)
    #     if i == 0:
    #         values = df[0].values
    #     else:
    #         values += df[0].values
    # values = values / 5
    # out_file = './result_reverse.csv'
    # F = open(out_file, 'w')
    # for p in values:
    #     F.write('%f\n' % p)
    # F.close()
