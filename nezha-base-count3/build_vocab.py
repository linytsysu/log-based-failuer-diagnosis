import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

log_df = pd.read_csv('../new_src/log_template.csv')

data = []
for i in tqdm(range(log_df.shape[0])):
    msg = log_df.iloc[i]['msg_lower']
    msg = msg.split(' ')
    msg = list(filter(lambda x: len(x) > 0, msg))
    data.append(msg)

token2count = Counter()
for i in data:
    token2count.update(i)

pre=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',]
tail = []
for k, v in token2count.items():
    if v >= 3:
        tail.append(k)
vocab = pre + tail
with open('./vocab.txt', "w", encoding="utf-8") as f:
    for i in vocab:
        f.write(str(i)+'\n')
