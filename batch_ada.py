# -*- coding: utf8 -*-
import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd

class Batch_ada(object):
    def __init__(self,learning_rate,batch_size,training_epochs):
        self.mnist_data = input_data.read_data_sets('./fashion-mnist/data/fashion',one_hot=True)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_epochs = training_epochs


    def init_model(self,nums):
        for i in range(nums):
            self.model_cnn(i)
            self.model_dnn(i)

    def model_dnn(self,index):
        x_tensor = tf.placeholder(tf.float32,[None,784])
        y_tensor = tf.placeholder(tf.float32, [None, 10])
        hidden_1 = tf.layers.dense(inputs=x_tensor,units=64,activation=tf.nn.relu,use_bias=True)
        hidden_2 = tf.layers.dense(inputs=hidden_1,units=32,activation=tf.nn.relu,use_bias=True)
        logits = tf.layers.dense(inputs=hidden_2,units=10,activation=None,use_bias=True)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_tensor,logits=logits)
        vars = tf.trainable_variables()
        vars_grad = tf.gradients(-loss,vars)
        vars_new=[]
        for var_grad,var in zip(vars_grad,vars):
            vars_new.append(tf.assign(var,var+self.learning_rate*var_grad))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.add_to_collection('x_tensor',x_tensor)
        tf.add_to_collection('y_tensor',y_tensor)
        tf.add_to_collection('loss',loss)
        for var_new in vars_new:
            tf.add_to_collection('vars_new',var_new)
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess,'./model_weight/batchsize_adaptive/model_dnn_{}'.format(index))
            batch_xs, batch_ys = self.mnist_data.train.next_batch(self.batch_size)
            loss_val = sess.run([loss], feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
            print(loss_val)
        tf.reset_default_graph()

    def model_cnn(self,index):
        x_tensor = tf.placeholder(tf.float32,[None,28,28,1])
        y_tensor = tf.placeholder(tf.float32, [None, 10])
        # 28*28
        temp = tf.layers.conv2d(inputs=x_tensor,filters=6,kernel_size=(5,5),use_bias=True)
        # 24*24
        temp = tf.layers.max_pooling2d(inputs=temp,pool_size=(2,2),strides=2)
        # 12*12
        temp = tf.layers.conv2d(inputs=temp,filters=16,kernel_size=(5,5),use_bias=True)
        # 8*8
        temp = tf.layers.max_pooling2d(inputs=temp,pool_size=(2,2),strides=2)
        # 4*4
        temp = tf.layers.conv2d(inputs=temp,filters=120,kernel_size=(4,4),use_bias=True)
        temp = tf.layers.flatten(inputs=temp)
        temp = tf.layers.dense(inputs=temp,units=84,activation=tf.nn.relu,use_bias=True)
        logits = tf.layers.dense(inputs=temp,units=10,activation=None,use_bias=True)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_tensor,logits=logits)
        vars = tf.trainable_variables()
        vars_grad = tf.gradients(-loss,vars)
        vars_new=[]
        for var_grad,var in zip(vars_grad,vars):
            vars_new.append(tf.assign(var,var+self.learning_rate*var_grad))
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        tf.add_to_collection('x_tensor',x_tensor)
        tf.add_to_collection('y_tensor',y_tensor)
        tf.add_to_collection('loss',loss)
        for var_new in vars_new:
            tf.add_to_collection('vars_new',var_new)
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess,'./model_weight/batchsize_adaptive/model_cnn_{}'.format(index))
        tf.reset_default_graph()

    def Batch_ada_train(self,index,batch_size_adaptive=False,flag_lst=[20,30,40],net='dnn'):
        if net == 'dnn':
            path = './model_weight/batchsize_adaptive/model_dnn_{}.meta'
        else:
            path = './model_weight/batchsize_adaptive/model_cnn_{}.meta'
        saver = tf.train.import_meta_graph(path.format(index))
        x_tensor = tf.get_collection('x_tensor')[0]
        y_tensor = tf.get_collection('y_tensor')[0]
        loss = tf.get_collection('loss')[0]
        vars_new = []
        for var in tf.get_collection('vars_new'):
            vars_new.append(var)

        with tf.Session() as sess:
            if net == 'dnn':
                saver.restore(sess,'./model_weight/batchsize_adaptive/model_dnn_{}'.format(index))
            else:
                saver.restore(sess,'./model_weight/batchsize_adaptive/model_cnn_{}'.format(index))
            total_batch = 0
            all_batch = 0
            flag = 1
            data = self.mnist_data
            total_num = data.train.num_examples
            lst_loss = []
            cost = 0
            batch_size = self.batch_size
            while True:
                if batch_size_adaptive:
                    if flag == flag_lst[0]:
                        batch_size = 1024
                    if flag == flag_lst[1]:
                        batch_size = 2048
                    if flag == flag_lst[2]:
                        batch_size = 4096
                if flag>training_epochs:
                    break
                batch_xs, batch_ys = data.train.next_batch(batch_size)
                if net=='cnn':
                    batch_xs = batch_xs.reshape(-1,28,28,1)
                c = sess.run(vars_new+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                c = c[-1]
                total_batch+=batch_size
                all_batch+=batch_size
                cost += c * batch_size
                if int(all_batch / total_num) == flag:
                    if net == 'dnn':
                        loss_val = loss.eval({x_tensor: data.test.images, y_tensor: data.test.labels})
                    else:
                        loss_val = loss.eval({x_tensor: data.test.images.reshape(-1,28,28,1), y_tensor: data.test.labels})
                    print("Epoch:", '%04d' % (flag), "cost=", "{:.6f}".format(cost/total_batch),'test acc',loss_val)
                    lst_loss.append(loss_val)
                    cost = 0
                    total_batch = 0
                    flag+=1
        tf.reset_default_graph()
        return lst_loss



if __name__ == '__main__':
    learning_rate = 0.1
    batch_size = 64
    training_epochs = 18
    ins = Batch_ada(learning_rate,batch_size,training_epochs)
    ins.init_model(100)
    '''
    flag_lst = [8,12,15]
    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Batch_ada_train(index=index,net='cnn',batch_size_adaptive=True,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('cnn_batch_ada.csv',header=None,index=None)

    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Batch_ada_train(index=index,net='cnn',batch_size_adaptive=False,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('cnn_batch_data_no.csv',header=None,index=None)
    '''
    training_epochs = 50
    flag_lst = [20,30,40]
    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Batch_ada_train(index=index,net='dnn',batch_size_adaptive=False,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('dnn_batch_ada_no.csv',header=None,index=None)

    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Batch_ada_train(index=index,net='dnn',batch_size_adaptive=True,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('dnn_batch_ada.csv',header=None,index=None)
