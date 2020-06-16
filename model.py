import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import viterbi_decode


def network(inputs,shapes,num_tags,lstm_dim=100,initializer = tf.truncated_normal_initializer()):
    '''
    接收一个批次样本的特征数据，计算网络的输出值
    :param char: type of int, a tensor of shape 2-D [None,None]
    :param bound: a tensor of shape 2-D [None,None] with type of int
    :param flag: a tensor of shape 2-D [None,None] with type of int
    :param radical: a tensor of shape 2-D [None,None] with type of int
    :param pinyin: a tensor of shape 2-D [None,None] with type of int
    :return:
    '''

    # -----------------------------------特征嵌入-------------------------------------
    #将所有特征的id转换成一个固定长度的向量
    embedding=[]
    keys = list(shapes.keys())
    for key in keys:
        with tf.variable_scope(key+'_embedding'):
            char_lookup = tf.get_variable(
                name=key+'_embedding',
                shape=shapes[key],
                initializer=initializer
            )
            # 每一个char的id找到char_lookup对应饿行，即该字对应的向量
            embedding.append(tf.nn.embedding_lookup(char_lookup, inputs[key]))#实现特征的嵌入
    embed = tf.concat(embedding,axis=-1)#shape [None, None, char_dim+bound_dim+flag_dim+radical_dim+pinyin_dim]

    #拿到输入里面的字符数据,正数变成1，0变成0
    sign = tf.sign(tf.abs(inputs[keys[0]]))
    #得到每个句子真实的长度
    lengths = tf.reduce_sum(sign,reduction_indices = 1)
    #得到序列的长度
    num_time = tf.shape(inputs[keys[0]])[1]



    # --------------------------------循环神经网络编码--------------------------------
    with tf.variable_scope('BiLSTM_layer1'):
        lstm_cell = {}
        for name in ['forward1','backward1']:
            with tf.variable_scope(name):
                lstm_cell[name] = rnn.BasicLSTMCell(
                    #有多少个神经元是init指定好传过来的
                    lstm_dim

                )
        #双向的动态rnn，来回都是100，拼接起来是200
        outputs1,finial_states1 = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward1'],
            lstm_cell['backward1'],
            embed,
            dtype = tf.float32,
            #告知实际的长度
            sequence_length = lengths
        )
    outputs1 = tf.concat(outputs1,axis = -1) #b,L,2*lstm_dim

    with tf.variable_scope('BiLSTM_layer2'):
        lstm_cell = {}
        for name in ['forward','backward']:
            with tf.variable_scope(name):
                lstm_cell[name] = rnn.BasicLSTMCell(
                    #有多少个神经元是init指定好传过来的
                    lstm_dim
                )
        #双向的动态rnn，来回都是100，拼接起来是200
        outputs,finial_statesl = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell['forward'],
            lstm_cell['backward'],
            outputs1,
            dtype = tf.float32,
            #告知实际的长度
            sequence_length = lengths
        )
    output = tf.concat(outputs,axis = -1) #b,L,2*lstm_dim



    # --------------------------------输出映射--------------------------------
    #矩阵乘法只能是两维的
    #reshape成二维矩阵 batch_size*maxlength,2*lstm_dim
    output = tf.reshape(output,[-1,2*lstm_dim])
    with tf.variable_scope('project_layer1'):
        w = tf.get_variable(
            name = 'w',
            shape = [2*lstm_dim,lstm_dim],
            initializer = initializer
        )
        b = tf.get_variable(
            name = 'b',
            shape = [lstm_dim],
            initializer = tf.zeros_initializer()
        )
        output  =tf.nn.relu(tf.matmul(output,w)+b)
    with tf.variable_scope('project_layer2'):
        w = tf.get_variable(
            name = 'w',
            shape = [lstm_dim,num_tags],
            initializer = initializer
        )
        b = tf.get_variable(
            name = 'b',
            shape = [num_tags],
            initializer = tf.zeros_initializer()
        )
        output  =tf.matmul(output,w)+b
    output = tf.reshape(output,[-1,num_time,num_tags])
    #batch_size,max_length,num_tags
    return output,lengths



class Model(object):
    def __init__(self, dict,lr = 0.0001):
        # --------------------------------用到的参数值--------------------------------

        #可以选择读字典计算长度，也可以直接给出一个数字
        self.num_char = len(dict['word'][0])
        self.num_bound = len(dict['bound'][0])
        self.num_flag = len(dict['flag'][0])
        self.num_radical = len(dict['radical'][0])
        self.num_pinyin = len(dict['pinyin'][0])
        self.num_tags = len(dict['label'][0])
        #指定每一个字被映射为多少长度的向量
        self.char_dim = 100
        self.bound_dim = 20
        self.flag_dim = 50
        self.radical_dim = 50
        self.pinyin_dim = 50
        self.lstm_dim = 100
        self.lr = lr
        self.map = dict

        # -----------------------定义接受数据的placeholder----------------------------
        self.char_inputs = tf.placeholder(dtype = tf.int32,
                                          shape = [None,None],
                                          name = 'char_inputs')
        self.bound_inputs = tf.placeholder(dtype=tf.int32,
                                           shape=[None, None],
                                           name='bound_inputs')
        self.flag_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name='flag_inputs')
        self.radical_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name='radical_inputs')
        self.pinyin_inputs = tf.placeholder(dtype=tf.int32,
                                            shape=[None, None],
                                            name='pinyin_inputs')
        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name='targets')
        self.global_step = tf.Variable(0,trainable = False)#不需要训练，只是用来计算
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]

        # ------------------------------计算模型输出值-------------------------------
        self.logits,self.lengths = self.get_logits(self.char_inputs,
                                                   self.bound_inputs,
                                                   self.flag_inputs,
                                                   self.radical_inputs,
                                                   self.pinyin_inputs
                                                   )

        # ------------------------------计算损失-------------------------------
        self.cost = self.loss(self.logits,self.targets,self.lengths)

        # ----------------------------优化器优化-------------------------------
        #采用梯度截断技术
        with tf.variable_scope('optimizer'):
            opt = tf.train.AdamOptimizer(self.lr)
            grad_vars = opt.compute_gradients(self.cost)#计算出所有参数的导数
            clip_grad_vars = [[tf.clip_by_value(g,-5,5),v] for g,v in grad_vars]#得到截断之后的梯度
            self.train_op  =opt.apply_gradients(clip_grad_vars,self.global_step)#使用截断后的梯度对参数进行更新

        self.saver = tf.train.Saver(tf.global_variables(),max_to_keep = 5)


    def get_logits(self,char,bound,flag,radical,pinyin):
        '''
        接收一个批次样本的特征数据，计算网络的输出值
        :param char: type of int, a tensor of shape 2-D [None,None]
        :param bound: a tensor of shape 2-D [None,None] with type of int
        :param flag: a tensor of shape 2-D [None,None] with type of int
        :param radical: a tensor of shape 2-D [None,None] with type of int
        :param pinyin: a tensor of shape 2-D [None,None] with type of int
        :return: 3-d tensor  [batch_size,max_length,num_tags]
        '''
        shapes = {}
        #有多少个元素*每个元素的维度
        shapes['char']=[self.num_char,self.char_dim]
        shapes['bound']=[self.num_bound,self.bound_dim]
        shapes['flag']=[self.num_flag,self.flag_dim]
        shapes['radical']=[self.num_radical,self.radical_dim]
        shapes['pinyin']=[self.num_pinyin,self.pinyin_dim]
        inputs= {}
        inputs['char'] = char
        inputs['bound'] = bound
        inputs['flag'] = flag
        inputs['radical'] =radical
        inputs['pinyin'] = pinyin

        return network(inputs,shapes,num_tags=self.num_tags,lstm_dim=self.lstm_dim,initializer = tf.truncated_normal_initializer())


    def loss(self, output, targets, lengths, initializer=None):
        '''
        该函数的主要功能：计算损失
        :param output:
        :param targets:
        :param lengths:
        :param initializer:
        :return:
        '''
        b = tf.shape(lengths)[0]
        num_steps = tf.shape(output)[1]
        with tf.variable_scope('crf_loss'):
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1


            )
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)

            logits = tf.concat([output,pad_logits],axis = -1)
            logits = tf.concat([start_logits,logits],axis = 1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1
            )

            self.trans = tf.get_variable(
                name = 'trans',
                shape = [self.num_tags+1,self.num_tags+1],
                initializer = tf.truncated_normal_initializer()
            )
            log_likehood,self.trans = tf.contrib.crf.crf_log_likelihood(
                inputs = logits,
                tag_indices = targets,
                transition_params = self.trans,
                sequence_lengths = lengths
            )
            return tf.reduce_mean(-log_likehood)

    def run_step(self,sess,batch,istrain = True,istest=False):
        '''
        该函数的主要功能：判断是否为训练集，并且分批读入数据
        :param sess:
        :param batch:
        :param istrain:
        :return:
        '''
        if istrain:
            feed_dict = {
                self.char_inputs:batch[0],
                self.targets: batch[1],
                self.bound_inputs:batch[2],
                self.flag_inputs:batch[3],
                self.radical_inputs:batch[4],
                self.pinyin_inputs:batch[5]
             }
            _, loss = sess.run([self.train_op, self.cost],feed_dict = feed_dict)
            return loss
        elif istest:
            feed_dict = {
                self.char_inputs:batch[0],
                self.bound_inputs:batch[2],
                self.flag_inputs:batch[3],
                self.radical_inputs:batch[4],
                self.pinyin_inputs:batch[5],
             }
            logits,lengths = sess.run([self.logits,self.lengths],feed_dict = feed_dict)
            return logits,lengths
        else:
            feed_dict = {
                self.char_inputs: batch[0],
                self.bound_inputs: batch[1],
                self.flag_inputs: batch[2],
                self.radical_inputs: batch[3],
                self.pinyin_inputs: batch[4],
            }
            logits, lengths = sess.run([self.logits, self.lengths], feed_dict=feed_dict)
            return logits, lengths



    def decode(self, logits, lengths, matrix):
        '''
        该函数的主要功能：对测试集进行预测
        :param logits:
        :param lengths:
        :param matrix:
        :return: 解码出的id
        '''
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            # 只取有效字符的输出
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=-1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits,matrix)

            paths.append(path[1:])
        return paths

    def result_to_json(self,string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item

    def predict(self,sess,batch,istrain=False,istest=True):
        '''
        该函数的主要功能：进行实际的预测，并且展示字和每个字的标记
        :param sess:
        :param batch:
        :return:
        '''
        results = []
        items = []
        matrix = self.trans.eval()
        logits,lengths = self.run_step(sess,batch,istrain,istest)
        paths = self.decode(logits,lengths,matrix)
        chars = batch[0]
        judge = 0
        total_length = 0
        if istest:
            for i in range(len(paths)):
                #第i句话对应的真实的长度
                length = lengths[i]
                string = [self.map['word'][0][index] for index in chars[i][:length]]
                tags = [self.map['label'][0][index] for index in paths[i]]
                result = [k for k in zip(string,tags)]
                results.append(result)
                #计算准确率
                labels = batch[1]
                # print('path[{}]:{}'.format(i,paths[i]))
                # print('label[{}]:{}'.format(i,labels[i]))
                judge += sum(np.array([paths[i][index]==labels[i][index] for index in range(length)]).astype(int))
                total_length += length
            presicion = judge/total_length*100
            return results,presicion
        else:
            for i in range(len(paths)):
                # 第i句话对应的真实的长度
                length = lengths[i]
                string = [self.map['word'][0][index] for index in chars[i][:length]]
                tags = [self.map['label'][0][index] for index in paths[i]]
                result = [k for k in zip(string, tags)]
                results.append(result)
                print(result)
                items = self.result_to_json(string, tags)
            return results,items

