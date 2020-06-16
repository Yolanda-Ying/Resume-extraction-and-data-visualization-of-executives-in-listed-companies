import time

import tensorflow as tf
from data_utils import BatchManager,get_dict
from model import Model

batch_size = 50
dict_file = 'data/Prepare/dict.pkl'

def train():
    # -----------------------------------数据准备-------------------------------------
    train_manager = BatchManager(batch_size = 20, name = 'train')
    test_manager = BatchManager(batch_size = 100,name='test')

    # -----------------------------------读取字典-------------------------------------
    mapping_dict = get_dict(dict_file)

    # -----------------------------------搭建模型-------------------------------------
    model = Model(mapping_dict)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(5):
            j = 1
            for batch in train_manager.iter_batch(shuffle=True):
                start = time.time()
                loss = model.run_step(sess,batch)
                end = time.time()
                if j%5 == 0:
                    print('epoch:{},step:{}/{},loss:{},elapse:{},estimate:{}'.format(i+1,
                                                                                     j,
                                                                                     train_manager.len_data,
                                                                                     loss,
                                                                                     end-start,
                                                                                     (end-start)*(train_manager.len_data-j)))
                j+=1
            for batch in test_manager.iter_batch(shuffle=True):
                test_result = model.predict(sess,batch,istrain=False,istest=True)
                print('precision rate:{} %',test_result[1])


if __name__ == '__main__':
    train()