#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: zyw
@file: preprocessing.py
@time: 2018/11/15
"""
from collections import Counter
import pandas as pd
import numpy as np
import fasttext
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SVD import *


def get_w2v_model(text_list, w_size):
    sentences = [x.split() for x in text_list]
    model = word2vec.Word2Vec(sentences, sg=1, size=w_size, window=5, min_count=1)
    # model.save('sohu_news_w2v_model')
    return model

def get_fasttext_model(text_list, w_size):
    sentences = " ".join(text_list)
    with open('temp_for_fasttext.txt', 'w', encoding='utf-8-sig') as f:
        f.write(sentences)
    model = fasttext.skipgram('temp_for_fasttext.txt', 'fasttext_word_model', min_count=1, dim=w_size)
    return model

def embedding_x(df, x_col_name, max_sequence_len, vector_choice, stop_words_list=None, w2v_dim=100):
    print('开始对文本进行向量化...')
    if vector_choice == 'w2v':
        X_df = df[x_col_name]
        w2v_model = get_w2v_model(X_df.values.tolist(), w2v_dim)
        X_data = np.zeros((len(df), max_sequence_len, w2v_dim))
        for i, content_str in enumerate(X_df.values.tolist()):
            for j, word in enumerate(content_str.split()):
                X_data[i, j] = w2v_model[word]

        print('X_data.shape ', X_data.shape, '\n')
        X_data = X_data.reshape(X_data.shape[0], -1)
        print('X_data.shape 压缩后维度', X_data.shape, '\n')

    elif vector_choice == 'fasttext':
        X_df = df[x_col_name]
        fasttext_model = get_fasttext_model(X_df.values.tolist(), w2v_dim)
        X_data = np.zeros((len(df), max_sequence_len, w2v_dim))
        for i, content_str in enumerate(X_df.values.tolist()):
            for j, word in enumerate(content_str.split()):
                X_data[i, j] = fasttext_model[word]

        print('X_data.shape ', X_data.shape, '\n')
        X_data = X_data.reshape(X_data.shape[0], -1)
        print('X_data.shape 压缩后维度', X_data.shape, '\n')

    elif vector_choice == 'tf_idf':
        X_list = df[x_col_name].values.tolist()
        # vectorizer = TfidfVectorizer(stop_words=stop_words_list, max_features=50)
        vectorizer = TfidfVectorizer(stop_words=stop_words_list)
        X_data = vectorizer.fit_transform(X_list).toarray()

    return X_data


def embedding_label(df, label_col_name):
    # 先统计
    print('开始对标签进行向量化...')
    label_account = df.groupby([label_col_name]).size().reset_index()
    label_account.rename(columns={0: 'num'}, inplace=True)
    label_account.loc['sum'] = label_account.apply(lambda x: x.sum())
    label_account.loc['sum', label_col_name] = '总数'
    label_account['radio'] = label_account['num'].apply(lambda x: '%.1f%%' % (100*x/label_account.loc['sum']['num']))
    print('\n各标签下的数量统计\n{0}\n'.format(label_account))

    # 再处理
    y_label_list = df.groupby([label_col_name]).size().reset_index()[label_col_name].values.tolist()
    y_label_dict = {x: i for i, x in enumerate(y_label_list)}
    y_data = df[label_col_name].apply(lambda x: y_label_dict[x])

    return y_data


def get_eng_label_from_seperate_txt(txt_path, label_path, N_each_label, max_sequence_len, test_size, vector_choice,
                                    stop_words_list=None, w2v_dim=100):
    """
    对文本和标签分成2个txt格式的英文文本数据预处理
    :param txt_path:
    :param label_path:
    :param N_each_label:
    :return:
    """
    print('reading data label from seperate txt...')
    # 读取文本并切分成单条数据的list
    with open(txt_path, 'r', encoding='utf-8-sig') as f_txt:
        txt = f_txt.read().strip()
        txt_list = txt.split('\n')
        txt_list = [x.strip() for x in txt_list]

    # 读取标签并切分成单个标签的list
    with open(label_path) as f_label:
        label = f_label.read().strip()
        label_list = label.split('\n')

    # 检查数据&标签的额长度是否一致
    if len(txt_list) != len(label_list):
        print('error：文本标签数量不一致')
    print('文本数据总条数'.format(len(txt_list)))

    # 统计每个类别的分布
    label_count_dict = Counter(label_list)
    print('统计各个类别的数量分布,共有类别{0}个\n{1}\n'.format(len(label_count_dict), label_count_dict))

    # 每个类别都只保留指定数量
    df = pd.DataFrame({'txt': txt_list, 'label': label_list})
    df['word_num'] = df['txt'].apply(lambda x: x.count(' ') + 1)
    df = df[df['word_num'] <= max_sequence_len]     # 截取掉长度超限的
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle 数据
    partial_df = pd.DataFrame()
    for label in label_count_dict.keys():
        df_ = df[df['label'] == label][:N_each_label]
        partial_df = partial_df.append(df_)
    df = partial_df.sample(frac=1).reset_index(drop=True)
    print('\n筛选每类{0}个后共有数据{1}条：\n'.format(N_each_label, len(df)))

    # 统计每条数据的最大词数
    word_num_list = df['word_num'].values.tolist()
    word_num_list.sort(reverse=True)
    max_actural_sequence = word_num_list[0]
    print('单条数据最多的词数：{0}'.format(max_actural_sequence), '\n')

    # 向量化
    X_data = embedding_x(df, 'txt', max_sequence_len, vector_choice)
    # svder = CSVD(X_data)
    # X_data = svder.DimReduce(0.9)
    # sc = StandardScaler()     # 因为超内存，暂时不做标准化
    # sc.fit(X_data)
    # X_data = sc.transform(X_data)
    y_data = embedding_label(df, 'label')

    # 拆分train test
    train_x, test_x, train_y, test_y = train_test_split(X_data, y_data, test_size=test_size, random_state=0)
    print('训练文本的维度：{0}'.format(train_x.shape))
    # print('训练数据前1条\n', train_x.tolist()[0], ' ', len(train_x.tolist()[0]))
    print('训练标签的维度：{0}'.format(train_y.shape))
    # print('训练标签前2条\n', train_y.tolist()[0], '\n', train_y.tolist()[1])
    return train_x, test_x, train_y, test_y


if __name__ == '__main__':
    txt_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets.txt'
    label_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets_gnd.txt'
    get_eng_label_from_seperate_txt(txt_path, label_path, 1000, 40, 0.2, 'tf_idf',
                                    w2v_model_path=None)