# -*- coding: utf8 -*-
import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd

class Moment_ada(object):
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
        learning_rate = tf.placeholder(tf.float32,[])
        momentum = tf.placeholder(tf.float32,[])
        op_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(loss)
        tf.add_to_collection('x_tensor',x_tensor)
        tf.add_to_collection('y_tensor',y_tensor)
        tf.add_to_collection('loss',loss)
        tf.add_to_collection('learning_rate',learning_rate)
        tf.add_to_collection('momentum',momentum)
        tf.add_to_collection('op_train',op_train)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess,'./model_weight/moment_adaptive/model_dnn_{}'.format(index))
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
        learning_rate = tf.placeholder(tf.float32,[])
        momentum = tf.placeholder(tf.float32,[])
        #op_train = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(loss)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads = optimizer.compute_gradients(loss)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 100), v)  # clip gradients
                train_op = optimizer.apply_gradients(grads)

        #op_train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
        tf.add_to_collection('x_tensor',x_tensor)
        tf.add_to_collection('y_tensor',y_tensor)
        tf.add_to_collection('loss',loss)
        tf.add_to_collection('learning_rate',learning_rate)
        tf.add_to_collection('momentum',momentum)
        tf.add_to_collection('op_train',train_op)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess,'./model_weight/moment_adaptive/model_cnn_{}'.format(index))
        tf.reset_default_graph()


    def Moment_ada_train(self,index,moment_ada=False,flag_lst=[20,30],net='dnn'):
        if net == 'dnn':
            path = './model_weight/moment_adaptive/model_dnn_{}.meta'
        else:
            path = './model_weight/moment_adaptive/model_cnn_{}.meta'
        saver = tf.train.import_meta_graph(path.format(index))
        x_tensor = tf.get_collection('x_tensor')[0]
        y_tensor = tf.get_collection('y_tensor')[0]
        loss = tf.get_collection('loss')[0]
        learning_rate = tf.get_collection('learning_rate')[0]
        momentum = tf.get_collection('momentum')[0]
        op_train = tf.get_collection('op_train')[0]
        data = self.mnist_data
        display_step = 1
        lst_loss = []
        with tf.Session() as sess:
            if net == 'dnn':
                saver.restore(sess,'./model_weight/moment_adaptive/model_dnn_{}'.format(index))
            else:
                saver.restore(sess,'./model_weight/moment_adaptive/model_cnn_{}'.format(index))
            learning_rate_num = 0.1
            #momentum_num = 0.9
            momentum_num = 0.5
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(data.train.num_examples / self.batch_size)
                if epoch == flag_lst[0]:
                    learning_rate_num = 0.01
                    if moment_ada:
                        #momentum_num = 0.9684
                        momentum_num = 0.8418
                if epoch == flag_lst[1]:
                    learning_rate_num = 0.001
                    if moment_ada:
                        #momentum_num = 0.99
                        momentum_num = 0.95
                for i in range(total_batch):
                    batch_xs, batch_ys = data.train.next_batch(self.batch_size)
                    if net=='cnn':
                        batch_xs = batch_xs.reshape(-1,28,28,1)
                    c=sess.run([op_train,loss], feed_dict={x_tensor: batch_xs, y_tensor: batch_ys,learning_rate:learning_rate_num,momentum:momentum_num})
                    #c=sess.run([op_train,loss], feed_dict={x_tensor: batch_xs, y_tensor: batch_ys,learning_rate:learning_rate_num})
                    c = c[-1]
                    avg_cost += c / total_batch
                if (epoch + 1) % display_step == 0:
                    lst_loss.append(avg_cost)
                    print("Epoch:", '%04d' % (epoch), "train loss=", "{:.6f}".format(avg_cost))
        tf.reset_default_graph()
        return lst_loss


if __name__ == '__main__':
    learning_rate = 0.1
    batch_size = 64
    training_epochs = 50
    ins = Moment_ada(learning_rate,batch_size,training_epochs)
    #ins.init_model(100)
    '''
    flag_lst = [20,35]
    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Moment_ada_train(index=index,net='dnn',moment_ada=True,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('dnn_moment_ada.csv',header=None,index=None)

    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Moment_ada_train(index=index,net='dnn',moment_ada=False,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('dnn_moment_ada_no.csv',header=None,index=None)
    '''
    flag_lst = [20,35]
    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Moment_ada_train(index=index,net='cnn',moment_ada=True,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('cnn_moment_ada.csv',header=None,index=None)

    lst = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Moment_ada_train(index=index,net='cnn',moment_ada=False,flag_lst=flag_lst)
        lst.append(lst_loss)
    df_csv = pd.DataFrame(lst).T
    df_csv.to_csv('cnn_moment_ada_no.csv',header=None,index=None)
