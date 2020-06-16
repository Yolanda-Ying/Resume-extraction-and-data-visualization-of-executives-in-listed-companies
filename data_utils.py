import math
import os
import pickle
import random

import pandas as pd
from tqdm import tqdm


def get_data(name = 'train'):
    '''
    该函数的主要功能是：把所有的数据都放在一个文件里面一起获取，并且将数据进行不同形式的拼接，进行数据增强
    :param name:所有数据所在的位置
    :return:
    '''
    with open(f'data/Prepare/dict.pkl','rb') as f:
        map_dict = pickle.load(f)


    def item2id(data,w2i):
        '''
        该函数的主要功能是：把字符转变成id
        :param data: 等待转化的数据
        :param w2i: 转化的方法
        :return: 如果是认识的值就返回对应的ID，如果不认识，就返回UNK的id
        '''
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    results = []
    root = os.path.join('data/Prepare/',name)
    files = list(os.listdir(root))
    fileindex=-1
    file_index = []


    for file in tqdm(files):
        result=[]

        path = os.path.join(root,file)

        try:
            samples = pd.read_csv(path, sep=',', encoding='GBK')
        except UnicodeEncodeError:
            samples = pd.read_csv(path, sep=',', encoding='UTF-8',errors='ignore')
        except Exception as e:
            print(e)

        num_samples = len(samples)
        fileindex += num_samples
        file_index.append(fileindex)
        # 存储好每个句子开始的下标
        sep_index = [-1]+samples[samples['word']=='sep'].index.tolist()+[num_samples]#-1,20,40,50

        # -----------------------------获取句子并且将句子全部转换成id----------------------------
        for i in range(len(sep_index)-1):
            start = sep_index[i]+1
            end = sep_index[i+1]
            data = []
            for feature in samples.columns:
                #print(list(samples[feature])[start:end],map_dict[feature][1])
                try:
                    data.append(item2id(list(samples[feature])[start:end],map_dict[feature][1]))
                except:
                    print(item2id(list(samples[feature])[start:end],map_dict[feature][1]))
                #print(data)
            result.append(data)
        #按照数据进行不同的拼接，不拼接、拼接1个、拼接2个...从而增强数据学习的能力

        # ----------------------------------------数据增强-------------------------------------
        if name == 'task':
            results.extend(result)
        else:
            two=[]
            for i in range(len(result)-1):
                first = result[i]
                second = result[i+1]
                two.append([first[k]+second[k] for k in range(len(first))])

            three = []
            for i in range(len(result) - 2):
                first = result[i]
                second = result[i + 1]
                third = result[i + 2]
                three.append([first[k] + second[k]+third[k] for k in range(len(first))])
            #应该用extend而不是append
            results.extend(result+two+three)

    with open(f'data/Prepare/'+name+'.pkl','wb') as f:
        pickle.dump(results,f)


def get_dict(path):
    with open(path,'rb') as f:
        dict = pickle.load(f)
    return dict

class BatchManager(object):
    def __init__(self,batch_size,name='train'):
        with open(f'data/Prepare/' + name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        self.batch_data = self.sort_and_pad(data,batch_size,name)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self,data,batch_size,name):
        #总共有多少批次
        num_batch = int(math.ceil(len(data)/batch_size))
        #print(len(data[0][0]))
        #按照句子长度进行排序
        sorted_data = sorted(data,key=lambda x:len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size):(i+1)*int(batch_size)],name))
        return batch_data


    @staticmethod
    def pad_data(data,name):
        if name!='task':
            chars = []
            targets = []
            bounds = []
            flags = []
            radicals = []
            pinyins = []

            max_length = max([len(sentence[0]) for sentence in data])  # len(data[-1][0])
            for line in data:
                char, target, bound, flag, radical, pinyin = line
                padding = [0] * (max_length - len(char))
                chars.append(char + padding)
                targets.append(target + padding)
                bounds.append(bound + padding)
                flags.append(flag + padding)
                radicals.append(radical + padding)
                pinyins.append(pinyin + padding)
            return [chars,targets, bounds,flags,radicals,pinyins]
        else:
            chars = []
            bounds = []
            flags = []
            radicals = []
            pinyins = []

            max_length = max([len(sentence[0]) for sentence in data])  # len(data[-1][0])
            for line in data:
                char, bound, flag, radical, pinyin = line
                padding = [0] * (max_length - len(char))
                chars.append(char + padding)
                bounds.append(bound + padding)
                flags.append(flag + padding)
                radicals.append(radical + padding)
                pinyins.append(pinyin + padding)
            return [chars, bounds, flags, radicals, pinyins]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


if __name__ == '__main__':
    get_data('train')
    get_data('test')
    #get_data('task')

    # try:
    #     file_index.to_csv(save_path, index=False, encoding='GBK')
    # except UnicodeEncodeError:
    #     file_index.to_csv(save_path, index=False, encoding='UTF-8')
    # except Exception as e:
    #     print(e)

    #print(get_data('train'))
    #train_data = BatchManager(10,'train')