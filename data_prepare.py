import os
import pandas as pd
import pickle
from collections import Counter
from data_process import split_text
from tqdm import tqdm
import jieba.posseg as psg
from cnradical import Radical,RunOption
import shutil
from random import shuffle


train_dir = "data/Train"
task_dir = 'data/Test'

def process_text(idx,split_method = None,file_dir = 'train_dir',split_name = 'train'):
    '''
    该函数的主要功能：读取文本，进行切割，并且打上标记，提取词边界、词性、偏旁部首、拼音等文本特征
    :param idx: 文件名字，不含扩展名
    :param split_method: 切割文本的函数方法
    :param split_name: 最终保存的文件夹名字
    :return:
    '''
    #很庞大的字典，把每个字的各种属性都放进去
    data = {}

    #----------------------------------获取句子----------------------------------------
    #获取句子，如果制定了分割方法，就用分割方法返回当前句子分割完的句子组；如果没有指定就一行一行读
    if split_name == 'task':
        try:
            with open(f'data/Test/need_annotation.txt', 'r', encoding='UTF-8', errors='ignore') as f:
                texts = f.readlines()
        except UnicodeEncodeError:
            with open(f'data/Test/need_annotation.txt', 'r', encoding='GBK', errors='ignore') as f:
                texts = f.readlines()
        except Exception as e:
            print(e)

    else:
        if split_method is None:
            try:
                with open(f'{file_dir}/{idx}.txt','r',encoding='GBK') as f:
                    texts = f.readlines()
            except UnicodeEncodeError:
                with open(f'{file_dir}/{idx}.txt','r',encoding='UTF-8',errors='ignore') as f:
                    texts = f.readlines()
            except Exception as e:
                print(e)

        else:
            try:
                with open(f'{file_dir}/{idx}.txt','r',encoding='GBK') as f:
                    texts = f.read()
                    texts = split_method(texts)
            except UnicodeEncodeError:
                with open(f'{file_dir}/{idx}.txt','r',encoding='UTF-8',errors='ignore') as f:
                    texts = f.read()
                    texts = split_method(texts)
            except Exception as e:
                print(e)
    data['word'] = texts


    #----------------------------------获取标签----------------------------------------
    # 获取标签，先初始化为'O'
    # 遍历每个句子，再遍历每个句子的每个字，每个字的tag初始化为'O'
    tag_list = ['O' for s in texts for x in s]

    def replace_l(index, char):  # 替换字符串第一个字符
        s = list(tag_list[index])
        s[0] = char
        tag_list[index] = "".join(s)

    def check_single(idx, cls):  # 检查单个标注时所在索引位置是否已经标注并做处理
        if idx < 2 or idx > len(tag_list) - 2:
            return
        if (tag_list[idx] != 'O'):
            tag_list[idx] = 'S-' + cls
            if tag_list[idx - 1] != 'O' and tag_list[idx - 2] != 'O':
                replace_l(idx - 1, 'E')
            elif tag_list[idx - 1] != 'O':
                replace_l(idx - 1, 'S')

            if tag_list[idx + 1] != 'O' and tag_list[idx + 2][1] != 'O':
                replace_l(idx + 1, 'B')
            elif tag_list[idx + 1] != 'O':
                replace_l(idx + 1, 'S')
        else:
            tag_list[idx] = 'S-' + cls

    def check(start, end, cls):  # 检查多个字标注时开始位置和结束位置是否已经标注并处理
        if tag_list[start] != 'O' and start > 2:
            if tag_list[start - 1] != 'O' and tag_list[start - 2] != 'O':
                replace_l(start - 1, 'E')
            elif tag_list[start - 1] != 'O':
                replace_l(start - 1, 'S')
        if tag_list[end] != 'O' and end < len(tag_list) - 2:
            if tag_list[end + 1] != 'O' and tag_list[end + 2] != 'O':
                replace_l(end + 1, 'B')
            elif tag_list[end + 1] != 'O':
                replace_l(end + 1, 'S')

    if file_dir == train_dir:
        #从ann文件中获取标签
        tag = pd.read_csv(f'{file_dir}/{idx}.ann',header=None,sep= '\t')
        for i in range(tag.shape[0]):
            tag_item = tag.iloc[i][1].split(' ')
            #有些短标签中有空格，所以想要获取标签的终止位置最好用-1
            cls,start,end = tag_item[0],int(tag_item[1]),int(tag_item[-1])
            #一定要注意start的表示标签对应的第一个字符，但是end对应的是标签最后一个字符的后一个字符
            if end - start == 1:
                check_single(start, cls)
                continue
            check(start, end - 1, cls)
            tag_list[start] = 'B-'+cls
            for j in range(start+1,end):
                tag_list[j] = 'I-'+cls
            tag_list[end-1] = 'E-' + cls

        assert len([x for s in texts for x in s])==len(tag_list)


        text_list = ''
        for t in texts:
            text_list+=t
        textes = []
        tags = []
        start = 0
        end = 0
        max=len(tag_list)
        for s in texts:
            l = len(s)
            end += l
            if  end>=max or tag_list[end][0] != 'I':
                textes.append(text_list[start:end])
                tags.append(tag_list[start:end])
                start=end
        data['word']=textes
        data['label']=tags
        assert len([x for s in textes for x in s]) == len(tag_list)
    else:
        text_list = ''
        for t in texts:
            text_list += t
        textes = []
        start = 0
        end = 0
        max = len(tag_list)
        for s in texts:
            l = len(s)
            end += l
            if end >= max or tag_list[end][0] != 'I':
                textes.append(text_list[start:end])
                start = end
        data['word'] = textes
        assert len([x for s in textes for x in s]) == len(tag_list)



    #-----------------------------提取词性和词边界特征----------------------------------
    word_bounds=['M' for item in tag_list]#首先给所有的字都表上B标记
    word_flags=[]#用来保存每个字的词性特征
    for text in textes:
        for word,flag in psg.cut(text):
            if len(word)==1:#判断是一个字的词
                start=len(word_flags)#拿到起始下标
                word_bounds[start]='S'#标记修改为S
                word_flags.append(flag)#将当前词的词性名加入到wordflags列表
            else:
                start=len(word_flags)#获取起始下标
                word_bounds[start]='B'#第一个字打上B
                word_flags+=[flag]*len(word)#将这个词的每个字都加上词性标记
                end=len(word_flags)-1#拿到这个词的最后一个字的下标
                word_bounds[end]='E'#将最后一个字打上E标记


    #--------------------------------------统一截断---------------------------------------
    bounds = []
    flags=[]
    start = 0
    end = 0
    for s in textes:
        l = len(s)
        end += l
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        start += l
    data['bound'] = bounds
    data['flag']=flags

    # ----------------------------------------获取拼音特征-------------------------------------
    radical = Radical(RunOption.Radical)  # 提取偏旁部首
    pinyin = Radical(RunOption.Pinyin)  # 用来提取拼音
    # 提取偏旁部首特征  对于没有偏旁部首的字标上UNK
    data['radical'] = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in textes]
    # 提取拼音特征  对于没有拼音的字标上UNK
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in textes]



    # ------------------------------------------存储数据------------------------------------------------
    num_samples = len(textes)  # 获取有多少句话  等于是有多少个样本
    num_col = len(data.keys())  # 获取特征的个数 也就是列数

    dataset = []
    for i in range(num_samples):
        #用*转置
        records = list(zip(*[list(v[i]) for v in data.values()]))  # 解压
        dataset += records + [['sep'] * num_col]  # 每存完一个句子需要一行sep进行隔离
    dataset = dataset[:-1]  # 最后一行sep不要
    dataset = pd.DataFrame(dataset, columns=data.keys())  # 转换成dataframe

    if split_name == 'task':
        save_path = f'data/Prepare/task/need_annotation.csv'
    else:
        save_path = f'data/Prepare/{split_name}/{idx}.csv'

    #把很多的换行符处理一下，现在已经把标签都处理好了，可以修改原来的文本序列了
    def clean_word(w):
        if w=='\n':
            return 'LB'
        if w in [' ','\t','\u2003']:
            return 'SPACE'
        if w.isdigit():#将所有的数字都变成一种符号，数字本身其实没有意义，告诉算法就是一个数据就行了
            return 'num'
        return w
    dataset['word']=dataset['word'].apply(clean_word)

    try:
        dataset.to_csv(save_path, index=False, encoding='GBK')
    except UnicodeEncodeError:
        dataset.to_csv(save_path, index=False, encoding='UTF-8')
    except Exception as e:
        print(e)

def multi_process(split_method = None,train_ratio=0.8):
    '''
    该函数的主要功能：创建准备好进行训练和测试的文件夹，以及把现在已经准备好的Train文件夹中的数据全部都处理完，按照2：8分割好放入对应的文件夹中
    :param split_method: 用什么方式进行分割
    :param train_ratio: 训练数据占总数据的比例
    :return:
    '''
    #单纯创建只能创建两层，用shutil可以创建多层
    if os.path.exists('data/Prepare/'):
        shutil.rmtree('data/Prepare/')
    if not os.path.exists('data/Prepare/train/'):
        os.makedirs('data/Prepare/train')
        os.makedirs('data/prepare/test')
        os.makedirs('data/Prepare/task')
    idxs=list(set([ file.split('.')[0] for file in os.listdir(train_dir)]))#获取所有文件的名字，去掉后缀名
    for x in idxs:
        if x == '':
            idxs.remove('')


    shuffle(idxs)#打乱顺序
    index=int(len(idxs)*train_ratio)#拿到训练集的截止下标
    train_ids=idxs[:index]#训练集文件名集合
    test_ids=idxs[index:]#测试集文件名集合

    import multiprocessing as mp
    num_cpus=mp.cpu_count()#获取机器cpu的个数
    pool=mp.Pool(num_cpus)
    results=[]
    for idx in train_ids:
        result=pool.apply_async(process_text,args=(idx,split_method,train_dir,'train'))
        results.append(result)
    for idx in test_ids:
        result=pool.apply_async(process_text,args=(idx,split_method,train_dir,'test'))
        results.append(result)
    pool.close()
    pool.join()
    [r.get() for r in results]

def task_process(split_method = None):
    '''
    该函数的主要功能是：将需要标注的任务文本进行处理，并且存储为csv格式的文件
    :param split_method:
    :return:
    '''
    idxs2 = list(set([ file.split('.')[0] for file in os.listdir(task_dir)]))
    for x in idxs2:
        if x == '':
            idxs2.remove('')
    task_ids=idxs2

    process_text(task_ids, split_method=None, file_dir='task_dir', split_name='task')



def mapping(data,threshold=0,is_word=False,sep='sep',is_label=False):
    '''
    该函数的主要功能：给每个标签按照从小到大的顺序排序后赋上序号
    注意：PAD做填充用，如果一开始按照句子长度排序，可以让每一批次的长度差不多，从而提高效率
    :param data:待链接下标的数据
    :param threshold:小于多少时视为为登陆词，需要去掉
    :param is_word:默认是否为词是false的
    :param sep:分割用'sep'
    :param is_label:默认是否为标签是false的
    :return:
    '''
    count=Counter(data)
    if sep is not None:
        count.pop(sep)
    if is_word:
        count['PAD']=100000001
        count['UNK']=100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data=[ x[0]  for x in data if x[1]>=threshold]#去掉频率小于threshold的元素，未登陆词。主动去掉频率比较少的词，模拟真实情况
        id2item=data
        item2id={id2item[i]:i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item,item2id


def get_dict():
    '''
    该函数的主要功能：遍历所有准备好的文件，找出所有关键词包含的值
    :return:没有返回值，字典字节存入data/Prepare/dict.pkl
    '''
    map_dict={}
    from glob import glob
    all_w,all_bound,all_flag,all_label,all_radical,all_pinyin=[],[],[],[],[],[]
    #glob用于遍历文件
    for file in glob('data/Prepare/train/*.csv')+glob('data/Prepare/test/*.csv'):
        try:
            df = pd.read_csv(file, sep=',', encoding='GBK')
        except UnicodeEncodeError:
            df = pd.read_csv(file, sep=',', encoding='UTF-8', errors='ignore')
        except Exception as e:
            print(e)

        all_w+=df['word'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()
        all_label += df['label'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    all_w.append('UNK')
    all_bound.append('UNK')
    all_flag.append('UNK')
    all_label.append('UNK')
    map_dict['word']=mapping(all_w,threshold=0,is_word=True)
    map_dict['bound']=mapping(all_bound)
    map_dict['flag']=mapping(all_flag)
    map_dict['label']=mapping(all_label,is_label=True)
    map_dict['radical']=mapping(all_radical)
    map_dict['pinyin']=mapping(all_pinyin)

    with open(f'data/Prepare/dict.pkl','wb') as f:
        pickle.dump(map_dict,f)




if __name__ == '__main__':
    multi_process(split_text)
    #task_process(split_text)
    get_dict()
    # with open(f'data/Prepare/dict.pkl','rb') as f:
    #     data = pickle.load(f)
    # print(data)
