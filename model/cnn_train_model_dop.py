import tensorflow as tf
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.compat.v1 import ConfigProto

def pre_process():
    label1 = np.load("D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/radar_pos_label_11-36_deleted.npy")
    label2 = np.load("D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/radar_pos_label_37-59_deleted.npy")
    deleted1 = np.load("D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/deleted_index_11-36.npy")
    deleted2 = np.load("D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/deleted_index_37-59.npy")
    deleted2 += 11664
    # print(deleted1.shape, deleted2.shape)
    deletedCom = np.concatenate((deleted1,deleted2))
    

    radar_data = np.load("D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/radar_data_reduce/radar_data_reduce_all_real_imag.npy")
    print(radar_data.shape)
    radar_data = np.swapaxes(np.swapaxes(radar_data, 1,2),2,3)
    # radar_data = radar_data[:,:,0,:]
    radar_data = np.delete(radar_data, deletedCom, axis =0)

    # radar_test_data = np.concatenate((radar_data[:486], radar_data[6317:6803], radar_data[12636:13122], radar_data[16038:16524], radar_data[18468:18954]))
    radar_test_data = np.concatenate((radar_data[6317:6803], radar_data[12636:13122], radar_data[16038:16524], radar_data[18468:18954]))
    # radar_train_data = np.concatenate((radar_data[486:6317], radar_data[6803:12636], radar_data[13122:16038], radar_data[16524:18468], radar_data[18954:]))
    radar_train_data = np.concatenate((radar_data[:6317], radar_data[6803:12636], radar_data[13122:16038], radar_data[16524:18468], radar_data[18954:]))
    print(radar_test_data.shape, radar_train_data.shape)

    dis_label = np.concatenate((label1, label2))
    dis_label = dis_label[:,0,:]
    # print(dis_label.shape)

    # dis_test_label = np.concatenate((dis_label[:486], dis_label[6317:6803], dis_label[12636:13122], dis_label[16038:16524], dis_label[18468:18954]))
    dis_test_label = np.concatenate((dis_label[6317:6803], dis_label[12636:13122], dis_label[16038:16524], dis_label[18468:18954]))
    # dis_train_label = np.concatenate((dis_label[486:6317], dis_label[6803:12636], dis_label[13122:16038], dis_label[16524:18468], dis_label[18954:]))
    dis_train_label = np.concatenate((dis_label[:6317], dis_label[6803:12636], dis_label[13122:16038], dis_label[16524:18468], dis_label[18954:]))
    print(dis_test_label.shape, dis_train_label.shape)
    # print("radar_data_shape = " ,radar_data.shape, "label_data_shape = ", dis_label.shape)
    
    return radar_train_data, radar_test_data, dis_train_label, dis_test_label


def fully_connected_layer(input1, input_channel, output_channel, name = "fc", activate = "leaky", is_training=False):
    
    with tf.variable_scope(name):
        w = tf.get_variable("Wf", shape=[input_channel, output_channel], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("B", shape = [output_channel], initializer=tf.contrib.layers.xavier_initializer())
    
    if activate == "leaky":
        op = tf.nn.leaky_relu(tf.matmul(input1, w) + b)
        op = tf.contrib.layers.batch_norm(op, is_training = is_training)
    else:
        op = tf.matmul(input1, w) + b

    # op = tf.contrib.layers.batch_norm(op, is_training = is_training)

    return op

def conv2d_layer(input1, stride_nn, filter_size, input_channel, number_filter, name = "conv2" , activate = "leaky", is_training=False):
    
    with tf.variable_scope(name):
        w = tf.get_variable("W2d", shape = [filter_size, filter_size, input_channel, number_filter], initializer= tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("B", shape = [number_filter], initializer = tf.contrib.layers.xavier_initializer())    
    conv2 = tf.nn.conv2d(input1, w, strides = [1, stride_nn, stride_nn, 1], padding="SAME")

    if activate == "leaky":
        op = tf.nn.leaky_relu(conv2 + b)
        op = tf.contrib.layers.batch_norm(op, is_training=is_training)
    else:
        op = conv2 + b
        # op = tf.layers.batch_normalization(op, training=training)
    return op

def L2_loss(denseOut, dis_label):

    L2_dis = tf.square(dis_label - denseOut)

    return tf.sqrt(tf.reduce_mean(L2_dis)), L2_dis

def run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label):

    tf.reset_default_graph()
 
    # learning_rate = 0.01
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.0098
    
    training_epoch = 10000
    batch_size = 1500
    log_dir = './summary_raw_data/summary_cnn_6_layers_batchnorm_complex_exp_2_reduce_test/2dcov2'
    model_path = './saved_model/model_cnn_6_layers_batchnorm_complex_exp_2_reduce_test/4plot.ckpt'
    # model_load = './saved_model/model_cnn_6_layers_batchnorm_complex_exp_1/4plot.ckpt'

    batch_test_size = 1944
    batch_test_len = 0

    
    x_p = tf.placeholder(tf.float32, shape=[None, 20, 20, 8], name= 'x_train')
    y_p = tf.placeholder(tf.float32, shape=[None, 3], name= 'y_train')
    training_phase = tf.placeholder(tf.bool, name = 'training_phase')


    conv2d = conv2d_layer(x_p, 1, 3, 8, 4, "conv2_layer1_en", activate="leaky", is_training = False)
    conv2d = conv2d_layer(conv2d, 2, 3, 4, 4, "conv2_layer2_en", activate="leaky", is_training = training_phase) 

    conv2d = conv2d_layer(conv2d, 1 , 3, 4, 8, "conv2_layer3_en", activate="leaky", is_training = False)
    conv2d = conv2d_layer(conv2d, 2 , 3, 8, 8, "conv2_layer4_en", activate="leaky", is_training = training_phase)

    conv2d = conv2d_layer(conv2d, 1 , 3, 8, 16, "conv2_layer5_en", activate="leaky", is_training= False)
    conv2d = conv2d_layer(conv2d, 2 , 3, 16, 16, "conv2_layer6_en", activate= "leaky", is_training= training_phase)

    # conv1d = conv2d_layer(conv1d, 1 , 3, 32, 32, "conv1_layer7_en", "leaky")
    # conv1d = conv2d_layer(conv1d, 2 , 3, 32, 64, "conv1_layer8_en", "leaky")    

    flatten = tf.reshape(conv2d, [-1, 3*3*16 ])

    dense1 = fully_connected_layer(flatten, 3*3*16, 200, "fc1", activate="leaky", is_training = training_phase)
    denseOut = fully_connected_layer(dense1, 200, 3, "fc2", activate="None", is_training = False)
    # denseOut= fully_connected_layer(dense2, 200, 3, "fcOut", "None")
    loss, hist = L2_loss(denseOut, y_p)


    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 12000, 0.96, staircase=True)

    train_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    # train_ops = tf.group([train_ops,update_ops])
    
    with tf.control_dependencies(update_ops):
        train_ops = train_optimiser.minimize(loss, global_step=global_step)

    hist_ops = tf.summary.histogram("histrogram_l2", hist)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        
        sess.run(init)
        
        writer1 = tf.summary.FileWriter(log_dir+"/training")
        writer2 = tf.summary.FileWriter(log_dir+"/validation")
        
        '''
        restore model path
        '''
        # saver.restore(sess, model_load)
        # print("model restore from file: %s" % model_load)


        for epoch in range(training_epoch):
            avg_loss = 0
            batch_len = 0
            total_batch = int(len(radar_train_data)/batch_size)
            
            for i in range(total_batch):
                x,c = sess.run([train_ops, loss], feed_dict= {x_p : radar_train_data[batch_len:(batch_len+batch_size)], y_p: dis_train_label[batch_len:(batch_len+batch_size)], training_phase: True})
                batch_len += batch_size
                avg_loss += c / total_batch


            training_acc = avg_loss
            test_acc, histrogram, lr = sess.run([loss, hist_ops, learning_rate], feed_dict = {x_p: radar_test_data[batch_test_len:(batch_test_len+batch_test_size)], y_p: dis_test_label[batch_test_len:(batch_test_len+batch_test_size)], training_phase: False})
            
            batch_test_len += batch_test_size
            if batch_test_len >= len(radar_test_data):
                batch_test_len = 0
                print("---- reset BTL to zero -----")
                print("----------------------------")

            print("epoch = ", epoch+1, "L2_distance_loss = ", training_acc)
            print("distance error = ", test_acc)
            print("-----------------------------------------------------------")


            summary_train = tf.Summary()
            summary_train.value.add(tag = "ACC1", simple_value=training_acc)
            summary_train.value.add(tag = "Learning_rate", simple_value=lr)
            writer1.add_summary(summary_train, epoch)
            

            summary_test = tf.Summary()
            summary_test.value.add(tag = "ACC1", simple_value=test_acc)
            writer2.add_summary(summary_test, epoch)

            writer2.add_summary(histrogram,epoch)

            

            '''
            saving model path
            '''
            if (epoch+1)%1000 == 0:
                print("-----Save model----- epoch + 1")
                save_path = saver.save(sess, model_path)
                print("Model saved in file: %s" %save_path)


    writer1.close()
    writer2.close()

def main():
    radar_train_data, radar_test_data, dis_train_label, dis_test_label = pre_process()
    # plt.imshow(radar_train_data[0,:,:,0])
    # plt.show()
    # radar_train_data = radar_train_data[:15000]
    # dis_train_label = dis_train_label[:15000]
    run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label)

if __name__ == "__main__":
    main()