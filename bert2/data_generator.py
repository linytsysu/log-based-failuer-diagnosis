import numpy as np
from bert4keras.snippets import sequence_padding, DataGenerator
from dataset import tokens

def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(4)
            output_ids.append(i)
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(np.random.choice(len(tokens)) + 9)
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids

def sample_convert(text, label, random=False):
    """转换为MLM格式
    """
    text_ids = [tokens.get(t, 1) for t in text]
    if random:
        text_ids, out_ids = random_mask(text_ids)
    else:
        out_ids = [0] * len(text_ids)
    token_ids = [2] + text_ids + [3]
    segment_ids = [0] * len(token_ids)
    output_ids = [label + 5] + out_ids + [0]
    return token_ids, segment_ids, output_ids

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, label) in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, label, random
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []