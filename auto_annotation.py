import shutil
import time

import tensorflow as tf
from data_utils import BatchManager,get_dict,get_data
from model import Model
import pandas as pd
import json
from data_process import split_text
from data_prepare import task_process
from train import train
import os
import re


batch_size = 50
dict_file = 'data/Prepare/dict.pkl'
raw_data = 'data/RawData/stockinfo.csv'

def annotation():
    #train()
    #读入整体的csv文件
    try:
        whole_data = pd.read_csv(raw_data, sep=',', encoding='UTF-8')
    except UnicodeEncodeError:
        whole_data = pd.read_csv(raw_data, sep=',', encoding='GBK', errors='ignore')
    except Exception as e:
        print(e)

    row_num = whole_data.shape[0]
    print(row_num)

    # -----------------------------------读取字典-------------------------------------
    mapping_dict = get_dict(dict_file)

    # -----------------------------------搭建模型-------------------------------------
    model = Model(mapping_dict)

    list = ['Per', 'Com', 'Time', 'Job', 'Nat', 'Bir', 'Age', 'Gdr', 'Uni', 'Edu', 'Sch', 'Col', 'Maj', 'Zhi', 'Hon']
    feature_dataframe = pd.DataFrame(columns=list)

    #创建test文件夹，并遍历所有的数据进行预测
    for i in range(whole_data):
        # 单纯创建只能创建两层，用shutil可以创建多层
        if os.path.exists('data/Test'):
            shutil.rmtree('data/Test')
        if not os.path.exists('data/Test'):
            os.makedirs('data/Test')

        cur_data = whole_data['ManagerResume'][i]
        print(cur_data)

        filename = 'data/Test/need_annotation.txt'
        with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(cur_data)

        task_process(split_text)
        get_data('task')

        # -----------------------------------数据准备-------------------------------------
        task_manager = BatchManager(batch_size=1, name='task')

        # -----------------------------------搭建模型-------------------------------------

        item_T = {}
        item_T = pd.DataFrame(item_T)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(1):
                for batch in task_manager.iter_batch(shuffle=True):
                    task_result,item = model.predict(sess, batch, istrain=False, istest=False)
                    #item_Entity = pd.DataFrame(item['entities'])
                    #item_T = item_T.append(item_Entity)
                    item_T = pd.DataFrame(item['entities'])

                    #print('predict result:{} %', task_result)
                    print(item_T)
                    #num_samples = len(item)  # 获取有多少句话  等于是有多少个样本
                    #print(num_samples)

        # -------------------------------存储标注完的数据----------------------------------
        f_Key = {}

        for feature in list:
            l_type =[]
            for j in range(item_T.shape[0]):
                if(item_T['type'].iloc[j] == feature):
                    return_word = [item_T['word'].iloc[j]]
                    l_type = l_type+return_word
            f_Key.update({feature:l_type})

        feature_dataframe = feature_dataframe.append(f_Key,ignore_index=True)

    FinalResult = pd.concat([whole_data,feature_dataframe], axis=1)
    fpath = 'FinalResult.csv'
    pd.DataFrame(FinalResult).to_csv(fpath)


if __name__ == '__main__':
    annotation()