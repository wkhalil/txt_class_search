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
from gensim.models import word2vec
from preprocessing import *
import warnings


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


def partial_df(df, y_label, label_n):
    label_df = df.groupby([y_label]).size().reset_index()
    label_list = label_df[y_label].values.tolist()
    partial_df = pd.DataFrame()
    for label in label_list:
        df_ = df[df[y_label] == label][:label_n]
        partial_df = partial_df.append(df_)
    return partial_df


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    txt_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets.txt'
    label_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets_gnd.txt'
    model_save_file = None
    model_save = {}

    # test_classifiers = ['LR', 'RF', 'Bagging', 'AdaBoost', 'SVM', 'SVMCV']
    test_classifiers = ['LR', 'RF']
    classifiers = {
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                    'Bagging': bagging_classifier,
                    'AdaBoost': AdaBoostClassifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   }

    # 读入数据
    train_x, test_x, train_y, test_y = get_eng_label_from_seperate_txt(txt_path, label_path, 800, 40, 0.2, 'tf_idf')

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