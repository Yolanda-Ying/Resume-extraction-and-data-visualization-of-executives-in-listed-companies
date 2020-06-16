import os
import re

train_dir = "data/Train"

def get_entities(dir):
    '''
    该函数的主要功能：统计当前.ann标注文件中用到的标注实体的个数，以及每个实体出现的数量
    :param dir:训练集所在的根目录
    :return:实体的字典，存储的是实体的名称和出现的次数
    '''
    entities = {}
    files = os.listdir(dir)

    #set去重
    files = list(set([file.split('.')[0] for file in files]))
    for x in files:
        if x == '':
            files.remove('')

    for file in files:
        path = os.path.join(dir,file+'.ann')
        try:
            with open(path, 'r', encoding='GBK',errors='ignore') as f:
                for line in f.readlines():
                    name = line.split('\t')[1].split(' ')[0]
                    if name in entities:
                        entities[name] += 1
                    else:
                        entities[name] = 1

        except UnicodeEncodeError:
            with open(path, 'r', encoding='UTF-8') as f:
                for line in f.readlines():
                    name = line.split('\t')[1].split(' ')[0]
                    if name in entities:
                        entities[name] += 1
                    else:
                        entities[name] = 1
        except Exception as e:
            print(e)

    return entities

def get_labelencoder(entities):
    '''
    该函数的主要功能：产生BIOSE序列标注的标签，并且得到标签和下标的映射
    :param entities:实体名称和数量构成的字典
    :return:
    '''
    #把所有的实体按照出现顺序降序排列
    entities = sorted(entities.items(),key=lambda x:x[1],reverse=True)
    #抽取降序排列完成之后的实体名称信息
    entities = [x[0] for x in entities]
    id2label = []
    id2label.append('O')
    for entity in entities:
        id2label.append('S-'+entity)
        id2label.append('B-'+entity)
        id2label.append('I-'+entity)
        id2label.append('E-'+entity)

    label2id = {id2label[i]:i for i in range(len(id2label))}
    return id2label,label2id


def ischinese(char):
    if '\u4e00'<= char <='\u9fff':
        return True
    return False


def split_text(text):
    '''
    该函数的主要功能：对句子按照标点符号进行分割，并且根据文本本身的特征，进行数据预处理
    :param text: 等待被分割的句子
    :return: 切割点的坐标
    '''
    split_index = []

    #按照某些字符进行分割，主要是。
    pattern = ' |. |, |, |: |; |。'
    for m in re.finditer(pattern,text):
        #特殊符号所在的下标
        idx = m.span()[0]
        if text[idx]==' ':
          continue
        if text[idx+1]=='\n':
            continue
        #把标点符号后的第一个句子的下标返回
        split_index.append(idx+1)
    #把第一位和最后一位的index加入
    split_index = list(sorted(set([0,len(text)]+split_index)))

    #对于长文本进行处理，应对LSTM不擅长处理长文本的特点
    other_index = []
    for i in range(len(split_index)-1):
        begin = split_index[i]
        end = split_index[i+1]
        other_index.append(begin)
        if end-begin >150:
            for j in range(begin,end):
                #遍历长句子找换行符/空格/,/;，如果换行符/空格/,/;的位置距离长句子开始已经超过50个字了，就拆句子
                if(j+1-other_index[-1])>50:
                    if text[j] == '\n':
                        other_index.append(j+1)
                    if text[j] == ' ':
                        other_index.append(j+1)
                    if text[j] == ';':
                        other_index.append(j+1)
                    if text[j] == ',':
                        other_index.append(j+1)

    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    # 处理短句子
    i = 0
    temp_idx = []
    while i < len(split_index) - 1:  # 0 10 20 30 45
        b = split_index[i]
        e = split_index[i + 1]

        num_ch = 0
        num_en = 0
        if e - b < 15:
            for ch in text[b:e]:
                if ischinese(ch):
                    num_ch += 1
                elif ch.islower() or ch.isupper():
                    num_en += 1
                if num_ch + 0.5 * num_en > 10:  # 如果汉字加英文超过5个  则单独成为句子
                    temp_idx.append(b)
                    i += 1
                    break
            if num_ch + 0.5 * num_en <= 10:  # 如果汉字加英文不到5个  和后面一个句子合并
                temp_idx.append(b)
                i += 2
        else:
            temp_idx.append(b)
            i += 1
    split_index = list(sorted(set([0, len(text)] + temp_idx)))

    # # 统计分句长度
    # lens = [split_index[i + 1] - split_index[i] for i in range(len(split_index)-1)][:-1]
    # print(max(lens),min(lens))

    result = []
    for i in range(len(split_index) - 1):
        result.append(text[split_index[i]:split_index[i + 1]])
    return result

if __name__ == '__main__':
    # #输出总共有多少实体
    # #当标注量为70的时候，没有用到age，只有14种实体，29种标注
    # print(len(get_entities(train_dir)))
    # #输出每个实体的个数
    # print(get_entities(train_dir))

    entities = get_entities(train_dir)
    label = get_labelencoder(entities)

    files = os.listdir(train_dir)
    files = list(set([file.split('.')[0] for file in files]))
    for x in files:
        if x == '':
            files.remove('')

    path = os.path.join(train_dir,files[0]+'.txt')

    try:
        with open(path, 'r', encoding='GBK', errors='ignore') as f:
            text = f.read()
            print(text)
            print(split_text(text))

    except UnicodeEncodeError:
        with open(path, 'r', encoding='UTF-8') as f:
            text = f.read()
            print(text)
            print(split_text(text))

    except Exception as e:
        print(e)
