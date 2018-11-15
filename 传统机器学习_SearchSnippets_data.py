#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: zyw
@file: LJT_class.py
@time: 2018/11/06
"""
# encoding: utf-8
"""
Author:ljt
Time:2018/9/8 15:03
file:class.py
"""

import time
import jieba
from sklearn import metrics
import pickle as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from collections import Counter


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


def bagging_classifier(train_x, train_y):
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    base_model = RandomForestClassifier()
    model = BaggingClassifier(base_model, n_estimators=100, max_samples=0.3)
    model.fit(train_x, train_y)
    return model


def AdaBoostClassifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    base_model = RandomForestClassifier()
    model = AdaBoostClassifier(base_model, n_estimators=100)
    model.fit(train_x, train_y)
    return model


def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data(X_data, y_df, test_size):
    train_y = y_df[int(len(y_df))*(1-test_size):]
    train_x = X_data[int(X_data.shape[0])*(1-test_size):]
    test_y = y_df[:int(y_df.shape[0])*test_size]
    test_x = X_data[:int(len(X_data))*test_size]
    return train_x, train_y, test_x, test_y


def get_w2v_model(text_list, w_size):
    sentences = [x.split() for x in text_list]
    model = word2vec.Word2Vec(sentences, sg=1, size=w_size, window=5, min_count=1)
    # model.save('sohu_news_w2v_model')
    return model


def str_jieba_cut(complete_str):
    cut_str = jieba.cut(complete_str, cut_all=False)
    cut_str = ' '.join(cut_str).replace('，', '').replace('。', '').replace('？', '').replace('！', '').replace('/', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace(':', '').replace('…', '').replace('（', '')\
        .replace('）', '').replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '')\
        .replace('’', '').replace('【', '').replace('】', '').replace('[', '').replace(']', '').replace('（', '')\
        .replace('）', '').replace('(', '').replace(')', '')       # 去标点
    return cut_str


def embedding_X(df, x_label, max_sequence_len, w2v_model):
    dims = w2v_model.vector_size
    X_df = df[x_label]
    X_data = np.zeros((len(df), max_sequence_len, dims))
    for i, content_str in enumerate(X_df.values.tolist()):
        for j, word in enumerate(content_str.split()):
            X_data[i, j] = w2v_model[word]

    print('X_data.shape ', X_data.shape, '\n')
    X_data = X_data.reshape(X_data.shape[0], -1)
    print('X_data.shape 压缩后维度', X_data.shape, '\n')
    return X_data


def embedding_label(df, y_label):
    # 先统计
    label_account = df.groupby([y_label]).size().reset_index()
    label_account.rename(columns={0: 'num'}, inplace=True)
    label_account.loc['sum'] = label_account.apply(lambda x: x.sum())
    label_account.loc['sum', y_label] = '总数'
    label_account['radio'] = label_account['num'].apply(lambda x: '%.1f%%' % (100*x/label_account.loc['sum']['num']))
    print('\n\n各标签下的数量统计\n{0}\n'.format(label_account))

    # 再处理
    y_label_list = df.groupby([y_label]).size().reset_index()[y_label].values.tolist()
    y_label_dict = {x: i for i, x in enumerate(y_label_list)}
    y_df = df[y_label].apply(lambda x: y_label_dict[x])

    return y_df


def partial_df(df, y_label, label_n):
    label_df = df.groupby([y_label]).size().reset_index()
    label_list = label_df[y_label].values.tolist()
    partial_df = pd.DataFrame()
    for label in label_list:
        df_ = df[df[y_label] == label][:label_n]
        partial_df = partial_df.append(df_)
    return partial_df


if __name__ == '__main__':
    embedding_dims = 100
    txt_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets.txt'
    label_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets_gnd.txt'
    model_save_file = None
    model_save = {}

    # test_classifiers = ['LR', 'RF', 'Bagging', 'AdaBoost', 'SVM', 'SVMCV']
    test_classifiers = ['AdaBoost']
    classifiers = {
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                    'Bagging': bagging_classifier,
                    'AdaBoost': AdaBoostClassifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   }

    # 读入数据
    print('reading training and testing data...')
    with open(txt_path, 'r', encoding='utf-8-sig') as f_txt:
        news_txt = f_txt.read().strip()
        news_txt_list = news_txt.split('\n')
        print(len(news_txt_list))
        news_txt_list = [x.strip() for x in news_txt_list]
        # for i, sent in enumerate(news_txt_list[:100]):
        #     print('{0} {1}'.format(i, sent))

    with open(label_path) as f_label:
        news_label = f_label.read().strip()
        news_label_list = news_label.split('\n')

    if len(news_txt_list) != len(news_label_list):
        print('error：文本标签数量不一致')

    input_num = len(news_txt_list)
    print('输入数据量\n', input_num, '\n')

    # 只获取每个类别中1000个
    label_count_dict = Counter(news_label_list)
    print('统计各个类别的数量分布,共有类别{1}个\n{0}\n'.format(label_count_dict, len(label_count_dict)))

    df = pd.DataFrame({'txt': news_txt_list, 'label': news_label_list})
    df = df.sample(frac=1).reset_index(drop=True)      # 打乱数据
    df = partial_df(df, 'label', 1000)
    print('\n筛选每类1000个后：\n', len(df))

    df['word_num'] = df['txt'].apply(lambda x: x.count(' ')+1)
    print(df.iloc[:3])

    print('\nword2vec训练开始...\n')
    w2v_model = get_w2v_model(df['txt'].values.tolist(), embedding_dims)
    print('word2vec模型包含词个数： {0}'.format(len(w2v_model.wv.vocab)))
    print('word2vec前几个词； {0}\n'.format(list(w2v_model.wv.vocab)[:3]))

    word_num_list = df['word_num'].values.tolist()
    word_num_list.sort(reverse=True)
    max_sequence_length = word_num_list[0]
    print('单条数据最多的词数  {0}'.format(max_sequence_length), '\n')

    X_data = embedding_X(df, 'txt', max_sequence_length, w2v_model)

    y_df = embedding_label(df, 'label')
    print('y_df.shape ', y_df.shape, '\n')

    train_x, test_x, train_y, test_y = train_test_split(X_data, y_df, test_size=0.2, random_state=0)
    # x_train_stand = StandardScaler().fit_transform(train_x)

    for classifier in test_classifiers:
        print('\n\n******************* {0} ********************'.format(classifier))
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file is not None:
            model_save[classifier] = model
        # 评估模型  准确率 精确率 召回率
        average_type = None
        precision = metrics.precision_score(test_y, predict, average=average_type)
        recall = metrics.recall_score(test_y, predict, average=average_type)
        print('precision: \n{0}\n\nrecall: \n{1}'.format(precision, recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: \n{0}'.format(accuracy))

    if model_save_file is not None:
        pickle.dump(model_save, open(model_save_file, 'wb'))