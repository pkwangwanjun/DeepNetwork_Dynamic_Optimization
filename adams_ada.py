# -*- coding: utf8 -*-
import keras
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np

class Adams_ada(object):
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
        vars_copy = []
        vars_grad_1 = []
        vars_grad_2 = []
        vars_grad_3 = []
        for var in vars:
            vars_copy.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
            vars_grad_1.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
            vars_grad_2.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
            vars_grad_3.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
        vars_grad = tf.gradients(-loss,vars)
        vars_op_1 = []
        vars_op_2 = []
        vars_op_3 = []
        vars_grad_op_1 = []
        vars_grad_op_2 = []
        vars_grad_op_3 = []
        vars_adam_op = []
        vars_swap_op_21 = []
        vars_swap_op_32 = []
        vars_swap_op_43 = []
        vars_adam_pece_op = []
        vars_copy_op = []
        vars_sgd_op = []

        def initial1():
            for var,var1,var_grad in zip(vars,vars_grad_1,vars_grad):
                vars_op_1.append(tf.assign(var,var+self.learning_rate*var_grad))
                vars_grad_op_1.append(tf.assign(var1,var_grad))
            return vars_op_1+vars_grad_op_1

        def initial2():
            for var,var2,var_grad in zip(vars,vars_grad_2,vars_grad):
                vars_op_2.append(tf.assign(var,var+self.learning_rate*var_grad))
                vars_grad_op_2.append(tf.assign(var2,var_grad))
            return vars_op_2+vars_grad_op_2

        def initial3():
            for var,var3,var_grad in zip(vars,vars_grad_3,vars_grad):
                vars_op_3.append(tf.assign(var,var+self.learning_rate*var_grad))
                vars_grad_op_3.append(tf.assign(var3,var_grad))
            return vars_op_3+vars_grad_op_3

        def sgds():
            for var,var_grad in zip(vars,vars_grad):
                vars_sgd_op.append(tf.assign(var,var+self.learning_rate*var_grad))
            return vars_sgd_op

        def adams():
            for index,(var,var_copy,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_copy,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_copy_op.append(tf.assign(var_copy,var.value()))
                vars_adam_op.append(tf.assign(var,var+self.learning_rate*(55/24.*var_grad-59/24.*var_g3+37/24.*var_g2-9/24.*var_g1)))
            return vars_adam_op+vars_copy_op

        def swap1():
            for index,(var,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_swap_op_21.append(tf.assign(var_g1,var_g2.value()))
            return vars_swap_op_21

        def swap2():
            for index,(var,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_swap_op_32.append(tf.assign(var_g2,var_g3.value()))
            return vars_swap_op_32

        def swap3():
            for index,(var,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_swap_op_43.append(tf.assign(var_g3,var_grad))
            return vars_swap_op_43



        def adams_pece():
            vars_grad_hidden = tf.gradients(-loss,vars)
            for index,(var_copy,var_g1,var_g2,var_g3,var_g_hidden) in enumerate(zip(vars_copy,vars_grad_1,vars_grad_2,vars_grad_3,vars_grad_hidden)):
                vars_adam_pece_op.append(tf.assign(var_copy,var_copy+self.learning_rate*(9/24.*var_g_hidden+19/24.*var_g3-5/24.*var_g2+var_g1)))
            return vars_adam_pece_op



        out1 = initial1()
        out2 = initial2()
        out3 = initial3()
        sgd_op = sgds()
        out4 = adams()
        swap_op1 = swap1()
        swap_op2 = swap2()
        swap_op3 = swap3()
        out5 = adams_pece()
        saver = tf.train.Saver()
        tf.add_to_collection('x_tensor',x_tensor)
        tf.add_to_collection('y_tensor',y_tensor)
        tf.add_to_collection('loss',loss)
        for i in out1:
            tf.add_to_collection('out1',i)
        for i in out2:
            tf.add_to_collection('out2',i)
        for i in out3:
            tf.add_to_collection('out3',i)
        for i in sgd_op:
            tf.add_to_collection('sgd_op',i)
        for i in out4:
            tf.add_to_collection('out4',i)
        for i in swap_op1:
            tf.add_to_collection('swap_op1',i)
        for i in swap_op2:
            tf.add_to_collection('swap_op2',i)
        for i in swap_op3:
            tf.add_to_collection('swap_op3',i)
        for i in out5:
            tf.add_to_collection('out5',i)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess,'./model_weight/adams_adaptive/model_dnn_{}'.format(index))
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


        vars_copy = []
        vars_grad_1 = []
        vars_grad_2 = []
        vars_grad_3 = []
        for var in vars:
            vars_copy.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
            vars_grad_1.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
            vars_grad_2.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
            vars_grad_3.append(tf.Variable(np.ones(var.shape,dtype='float32'),trainable=False))
        vars_grad = tf.gradients(-loss,vars)
        vars_op_1 = []
        vars_op_2 = []
        vars_op_3 = []
        vars_grad_op_1 = []
        vars_grad_op_2 = []
        vars_grad_op_3 = []
        vars_adam_op = []
        vars_swap_op_21 = []
        vars_swap_op_32 = []
        vars_swap_op_43 = []
        vars_adam_pece_op = []
        vars_copy_op = []
        vars_sgd_op = []

        def initial1():
            for var,var1,var_grad in zip(vars,vars_grad_1,vars_grad):
                vars_op_1.append(tf.assign(var,var+self.learning_rate*var_grad))
                vars_grad_op_1.append(tf.assign(var1,var_grad))
            return vars_op_1+vars_grad_op_1

        def initial2():
            for var,var2,var_grad in zip(vars,vars_grad_2,vars_grad):
                vars_op_2.append(tf.assign(var,var+self.learning_rate*var_grad))
                vars_grad_op_2.append(tf.assign(var2,var_grad))
            return vars_op_2+vars_grad_op_2

        def initial3():
            for var,var3,var_grad in zip(vars,vars_grad_3,vars_grad):
                vars_op_3.append(tf.assign(var,var+self.learning_rate*var_grad))
                vars_grad_op_3.append(tf.assign(var3,var_grad))
            return vars_op_3+vars_grad_op_3

        def sgds():
            for var,var_grad in zip(vars,vars_grad):
                vars_sgd_op.append(tf.assign(var,var+self.learning_rate*var_grad))
            return vars_sgd_op

        def adams():
            for index,(var,var_copy,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_copy,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_copy_op.append(tf.assign(var_copy,var.value()))
                vars_adam_op.append(tf.assign(var,var+self.learning_rate*(55/24.*var_grad-59/24.*var_g3+37/24.*var_g2-9/24.*var_g1)))
            return vars_adam_op+vars_copy_op

        def swap1():
            for index,(var,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_swap_op_21.append(tf.assign(var_g1,var_g2.value()))
            return vars_swap_op_21

        def swap2():
            for index,(var,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_swap_op_32.append(tf.assign(var_g2,var_g3.value()))
            return vars_swap_op_32

        def swap3():
            for index,(var,var_grad,var_g1,var_g2,var_g3) in enumerate(zip(vars,vars_grad,vars_grad_1,vars_grad_2,vars_grad_3)):
                vars_swap_op_43.append(tf.assign(var_g3,var_grad))
            return vars_swap_op_43



        def adams_pece():
            vars_grad_hidden = tf.gradients(-loss,vars)
            for index,(var_copy,var_g1,var_g2,var_g3,var_g_hidden) in enumerate(zip(vars_copy,vars_grad_1,vars_grad_2,vars_grad_3,vars_grad_hidden)):
                vars_adam_pece_op.append(tf.assign(var_copy,var_copy+self.learning_rate*(9/24.*var_g_hidden+19/24.*var_g3-5/24.*var_g2+var_g1)))
            return vars_adam_pece_op

        out1 = initial1()
        out2 = initial2()
        out3 = initial3()
        sgd_op = sgds()
        out4 = adams()
        swap_op1 = swap1()
        swap_op2 = swap2()
        swap_op3 = swap3()
        out5 = adams_pece()
        saver = tf.train.Saver()
        tf.add_to_collection('x_tensor',x_tensor)
        tf.add_to_collection('y_tensor',y_tensor)
        tf.add_to_collection('loss',loss)
        for i in out1:
            tf.add_to_collection('out1',i)
        for i in out2:
            tf.add_to_collection('out2',i)
        for i in out3:
            tf.add_to_collection('out3',i)
        for i in sgd_op:
            tf.add_to_collection('sgd_op',i)
        for i in out4:
            tf.add_to_collection('out4',i)
        for i in swap_op1:
            tf.add_to_collection('swap_op1',i)
        for i in swap_op2:
            tf.add_to_collection('swap_op2',i)
        for i in swap_op3:
            tf.add_to_collection('swap_op3',i)
        for i in out5:
            tf.add_to_collection('out5',i)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.save(sess,'./model_weight/adams_adaptive/model_cnn_{}'.format(index))
        tf.reset_default_graph()

    def Adams_ada_train(self,index,adams_adaptive=False,flag_threhold=20,net='dnn'):
        if net == 'dnn':
            path = './model_weight/adams_adaptive/model_dnn_{}.meta'
        else:
            path = './model_weight/adams_adaptive/model_cnn_{}.meta'
        saver = tf.train.import_meta_graph(path.format(index))
        x_tensor = tf.get_collection('x_tensor')[0]
        y_tensor = tf.get_collection('y_tensor')[0]
        loss = tf.get_collection('loss')[0]
        out1 = tf.get_collection('out1')
        out2 = tf.get_collection('out2')
        out3 = tf.get_collection('out3')
        out4 = tf.get_collection('out4')
        out5 = tf.get_collection('out5')
        swap_op1 = tf.get_collection('swap_op1')
        swap_op2 = tf.get_collection('swap_op2')
        swap_op3 = tf.get_collection('swap_op3')
        sgd_op = tf.get_collection('sgd_op')

        with tf.Session() as sess:
            if net == 'dnn':
                saver.restore(sess,'./model_weight/adams_adaptive/model_dnn_{}'.format(index))
            else:
                saver.restore(sess,'./model_weight/adams_adaptive/model_cnn_{}'.format(index))
            flag = 0
            opt = None
            display_step = 1
            lst_train_loss = []
            lst_test_loss = []
            for epoch in range(self.training_epochs):
                avg_cost = 0
                data = self.mnist_data
                total_batch = int(data.train.num_examples / self.batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = data.train.next_batch(self.batch_size)
                    if net == 'cnn':
                        batch_xs = batch_xs.reshape(-1,28,28,1)
                    if flag == 0:
                        c = sess.run(out1+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                    elif flag == 1:
                        c = sess.run(out2+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                    elif flag == 2:
                        c = sess.run(out3+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                    elif flag>=3 and (epoch<=flag_threhold or not adams_adaptive):
                        opt = 'sgd'
                        c = sess.run(sgd_op+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        sess.run(swap_op1,feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        sess.run(swap_op2,feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        sess.run(swap_op3,feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                    elif epoch>flag_threhold and adams_adaptive:
                        opt = 'adams'
                        c = sess.run(out4+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        sess.run(swap_op1,feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        sess.run(swap_op2,feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        sess.run(swap_op3,feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                        c = sess.run(out5+[loss],feed_dict={x_tensor: batch_xs, y_tensor: batch_ys})
                    flag+=1
                    c = c[-1]
                    avg_cost += c / total_batch
                if (epoch + 1) % display_step == 0:
                    lst_train_loss.append(avg_cost)
                    if net =='dnn':
                        test_loss = loss.eval({x_tensor: data.test.images, y_tensor: data.test.labels})
                    if net =='cnn':
                        test_loss = loss.eval({x_tensor: data.test.images.reshape(-1,28,28,1), y_tensor: data.test.labels})
                    lst_test_loss.append(test_loss)
                    print("Epoch:", '%04d' % (epoch + 1), "train loss=", "{:.6f}".format(avg_cost),'test loss={:.6f}'.format(test_loss),'opt:{}'.format(opt))
        tf.reset_default_graph()
        return lst_train_loss,lst_test_loss

if __name__ == '__main__':
    learning_rate = 0.01
    batch_size = 128
    training_epochs = 50
    ins = Adams_ada(learning_rate,batch_size,training_epochs)
    #ins.init_model(100)
    '''
    train_loss = []
    test_loss = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Adams_ada_train(index=index,net='dnn',adams_adaptive=True)
        train_loss.append(lst_loss[0])
        test_loss.append(lst_loss[1])
    df_csv_train = pd.DataFrame(train_loss).T
    df_csv_test = pd.DataFrame(test_loss).T
    df_csv_train.to_csv('dnn_adams_ada_train.csv',header=None,index=None)
    df_csv_test.to_csv('dnn_adams_ada_test.csv',header=None,index=None)

    train_loss = []
    test_loss = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Adams_ada_train(index=index,net='cnn',adams_adaptive=True)
        train_loss.append(lst_loss[0])
        test_loss.append(lst_loss[1])
    df_csv_train = pd.DataFrame(train_loss).T
    df_csv_test = pd.DataFrame(test_loss).T
    df_csv_train.to_csv('cnn_adams_ada_train.csv',header=None,index=None)
    df_csv_test.to_csv('cnn_adams_ada_test.csv',header=None,index=None)

    '''
    train_loss = []
    test_loss = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Adams_ada_train(index=index,net='dnn',adams_adaptive=False)
        train_loss.append(lst_loss[0])
        test_loss.append(lst_loss[1])
    df_csv_train = pd.DataFrame(train_loss).T
    df_csv_test = pd.DataFrame(test_loss).T
    df_csv_train.to_csv('dnn_adams_ada_train_no.csv',header=None,index=None)
    df_csv_test.to_csv('dnn_adams_ada_test_no.csv',header=None,index=None)

    train_loss = []
    test_loss = []
    for index in range(100):
        print(index)
        print('=========')
        lst_loss = ins.Adams_ada_train(index=index,net='cnn',adams_adaptive=False)
        train_loss.append(lst_loss[0])
        test_loss.append(lst_loss[1])
    df_csv_train = pd.DataFrame(train_loss).T
    df_csv_test = pd.DataFrame(test_loss).T
    df_csv_train.to_csv('cnn_adams_ada_train_no.csv',header=None,index=None)
    df_csv_test.to_csv('cnn_adams_ada_test_no.csv',header=None,index=None)
