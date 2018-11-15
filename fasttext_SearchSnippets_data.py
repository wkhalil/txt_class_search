#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: zyw
@file: fasttext_SearchSnippets_data.py
@time: 2018/11/14
"""
import fasttext
import random


def data_to_fasttext_form(txt_path, label_path, train_path, test_path, test_rate):
    with open(txt_path, 'r', encoding='utf-8-sig') as f_txt:
        news_txt = f_txt.read().strip()
        news_txt_list = news_txt.split('\n')
        print(len(news_txt_list))
        news_txt_list = [x.strip() for x in news_txt_list]

    with open(label_path) as f_label:
        news_label = f_label.read().strip()
        news_label_list = news_label.split('\n')

    if len(news_txt_list) != len(news_label_list):
        print('error：文本标签数量不一致')

    txt_label_list = []
    for i in range(len(news_txt_list)):
        txt_label_list.append(news_txt_list[i] + '  __label__' + news_label_list[i])
    random.shuffle(txt_label_list)     # 打乱数据

    input_num = len(news_txt_list)
    test_num = int(input_num*test_rate)
    test_list = txt_label_list[:test_num]
    train_list = txt_label_list[test_num:]
    print('输入数据量:{0}, 训练数据量：{1}, 测试数据量：{2}\n'.format(input_num, input_num-test_num, test_num))

    f_train = open(train_path, 'w+', encoding='utf-8-sig')
    for i in train_list:
        f_train.write(i + '\n')

    f_test = open(test_path, 'w+', encoding='utf-8-sig')
    for i in test_list:
        f_test.write(i + '\n')

    print('转换数据完成')


def fasttext_train(train_path, test_path, model_path):
    classifier = fasttext.supervised(train_path, model_path)
    test_result = classifier.test(test_path)
    print('准确率： {0}'.format(test_result.precision))
    print('召回率： {0}'.format(test_result.recall))
    print('预测数据的数量： {0}'.format(test_result.nexamples))
    with open(test_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        total_num = len(lines)
        wrong_num = 0
        for line in lines:
            line = line.strip()
            sentence, actual_label = line.split('__label__')
            fast_label = classifier.predict([sentence])
            if fast_label[0][0] != actual_label:
                print('\n预测错误')
                print('实际标签：{0}， 错误标签：{1}'.format(actual_label, fast_label[0][0]))
                print('句子: {0}\n'.format(sentence))
                wrong_num += 1
        print('\n总的数量为{0}， 预测错误的数量为{1}'.format(total_num, wrong_num))


if __name__ == '__main__':
    txt_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets.txt'
    label_path = '/home/zhangyu9/桌面/STC2-master/dataset/SearchSnippets_gnd.txt'
    train_path = 'SearchSnippets_fasttext_train.txt'
    test_path = 'SearchSnippets_fasttext_test.txt'
    fasttext_model_path = 'fasttext_searchsnippets.model'

    data_to_fasttext_form(txt_path, label_path, train_path, test_path, 0.2)
    fasttext_train(train_path, test_path, fasttext_model_path)


#
#
# classifier = fasttext.supervised('./data/train.txt', 'model_weibo_cate')
#
# num_predict = 0
# num_correct = 0
# with open('./data/test.txt', 'r') as f:
#   lines = f.readlines()
#   for line in lines:
#     line = line.rstrip()
#     sentence, real_label = line.split('__label__')
#     plabel = classifier.predict([sentence])
#     num_predict += 1
#     # predict_label = plabel[0][0].encode('ascii', 'ignore')
#     predict_label = plabel[0][0]
#     if real_label == predict_label:
#       num_correct += 1
#     else:
#       print('WARN: Incorrect, real_label=' + real_label + ' predicted=' + predict_label)
#       print('      ' + sentence)
#
# print('Total # is ', num_predict, ' # corrected prediction is ', num_correct)