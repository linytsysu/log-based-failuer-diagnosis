import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

label1 = pd.read_csv('../data/preliminary_train_label_dataset.csv')
label2 = pd.read_csv('../data/preliminary_train_label_dataset_s.csv')
label_df = pd.concat([label1, label2]).reset_index(drop=True)
label_df = label_df.drop_duplicates().reset_index(drop=True)

submit_df = pd.read_csv('../data/preliminary_submit_dataset_b.csv')

log_df = pd.read_csv('./log_template.csv')

log_df['time'] = pd.to_datetime(log_df['time'])
label_df['fault_time'] = pd.to_datetime(label_df['fault_time'])
submit_df['fault_time'] = pd.to_datetime(submit_df['fault_time'])

log_df['time_ts'] = log_df["time"].values.astype(np.int64) // 10 ** 9
label_df['fault_time_ts'] = label_df["fault_time"].values.astype(np.int64) // 10 ** 9
submit_df['fault_time_ts'] = submit_df["fault_time"].values.astype(np.int64) // 10 ** 9

log_df['time_gap'] = log_df['time'].dt.ceil('2h')

sentences_list = list()
for info, group in log_df.groupby(['sn', 'time_gap']):
    group = group.sort_values(by='time')
    sentences_list.append("\n".join(group['msg_lower'].values))

sentences = list()
for s in sentences_list:
    sentences.append([w for w in s.split()])

w2v_model = Word2Vec(sentences, vector_size=64, window=3, min_count=2, sg=0, hs=1, seed=2022)

def bow_extractor(corpus, ngram_range=(1,1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

bow_vectorizer, bow_features = bow_extractor(sentences_list)
feature_names = bow_vectorizer.get_feature_names_out()

def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix

tfidf_trans, tfidf_features = tfidf_transformer(bow_features)

def tfidf_extractor(corpus, ngram_range=(1,1)):

    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

tfidf_vectorizer, tdidf_features = tfidf_extractor(sentences_list)

vocab=tfidf_vectorizer.vocabulary_

def tfidf_wtd_avg_word_vectors(words,tfidf_vector,tfidf_vocabulary,model,num_features):
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                   if tfidf_vocabulary.get(word) else 0 for word in words]
    word_tfidf_map = {word:tfidf_val for word,tfidf_val in zip(words,word_tfidfs)}
    feature_vector = np.zeros((num_features,),dtype='float64')
    vocabulary = set(model.wv.index_to_key)
    wts = 0
    for word in words:
        if word in vocabulary:
            word_vector = model.wv[word]
            weighted_word_vector = word_tfidf_map[word]*word_vector
            wts = wts+word_tfidf_map[word]
            feature_vector = np.add(feature_vector,weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector,wts)
    return feature_vector

def tfidf_weighted_averaged_word_vectorizer(corpus,tfidf_vectors,tfidf_vocabulary,model,num_features):
    docs_tfidfs = [(doc,doc_tfidf) for doc,doc_tfidf in zip(corpus,tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence,tfidf,tfidf_vocabulary,model,num_features)
              for tokenized_sentence,tfidf in docs_tfidfs]
    return np.array(features)

def make_dataset(dataset, data_type='train'):
    ret = []
    for idx in tqdm(range(dataset.shape[0])):
        row = dataset.iloc[idx]
        sn = row['sn']
        fault_time = row['fault_time']
        fault_time_ts = row['fault_time_ts']
        sub_log = log_df[(log_df['sn'] == sn) & (log_df['time_ts'] <= fault_time_ts)]
        sub_log = sub_log.sort_values(by='time')
        df_tmp1 = sub_log.tail(20)
        new_sentences_list = ['\n'.join(df_tmp1['msg_lower'].values)]
        new_sentences = ['\n'.join(df_tmp1['msg_lower'].values).split()]
        feature = tfidf_weighted_averaged_word_vectorizer(
            corpus=new_sentences,
            tfidf_vectors=tfidf_vectorizer.transform(new_sentences_list),
            tfidf_vocabulary=vocab,
            model=w2v_model,
            num_features=64
        )
        data = {
            'sn': sn,
            'fault_time': fault_time
        }
        for i in range(64):
            data['weighted_w2v_%d'%i] = feature[0][i]
        if data_type == 'train':
            data['label'] = row['label']
        ret.append(data)
    return ret

train = make_dataset(label_df, data_type='train')
df_train = pd.DataFrame(train)

test = make_dataset(submit_df, data_type='test')
df_test = pd.DataFrame(test)

df_train.to_csv('train_new.csv', index=False)
df_test.to_csv('test_new.csv', index=False)

# import pandas as pd
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances

# df_train = pd.read_csv('train.csv')
# df_test = pd.read_csv('test.csv')

# featues = ['w2v_%d'%i for i in range(64)]
# kmeans = KMeans(n_clusters=4, random_state=0).fit(df_train[featues].values)

# dist = cosine_distances(df_train[featues].values, kmeans.cluster_centers_)
# df_train['cluster_distance_0'] = dist[:, 0]
# df_train['cluster_distance_1'] = dist[:, 1]
# df_train['cluster_distance_2'] = dist[:, 2]
# df_train['cluster_distance_3'] = dist[:, 3]

# dist = cosine_distances(df_test[featues].values, kmeans.cluster_centers_)
# df_test['cluster_distance_0'] = dist[:, 0]
# df_test['cluster_distance_1'] = dist[:, 1]
# df_test['cluster_distance_2'] = dist[:, 2]
# df_test['cluster_distance_3'] = dist[:, 3]

# df_train.to_csv('train_new.csv', index=False)
# df_test.to_csv('test_new.csv', index=False)
