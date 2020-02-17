import tensorflow as tf
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.compat.v1 import ConfigProto

def pre_process():
    label1 = np.load("D:/NestData/9-24-2019-single-chirp-16bit/pos_process_new/radar_pos_label_11-36_deleted.npy")
    label2 = np.load("D:/NestData/9-24-2019-single-chirp-16bit/pos_process_new/radar_pos_label_37-59_deleted.npy")
    deleted1 = np.load("D:/NestData/9-24-2019-single-chirp-16bit/pos_process_new/deleted_index_11-36.npy")
    deleted2 = np.load("D:/NestData/9-24-2019-single-chirp-16bit/pos_process_new/deleted_index_37-59.npy")
    deleted2 += 11664
    deletedCom = np.concatenate((deleted1,deleted2))


    radar_data = np.load("D:/NestData/9-24-2019-single-chirp-16bit/pos_process_new/radar_data.npy")
    radar_data = np.swapaxes(np.swapaxes(radar_data, 0,1),1,3)
    radar_data = radar_data[:,:,0,:]
    radar_data = np.delete(radar_data, deletedCom, axis =0)

    radar_test_data = np.concatenate((radar_data[:486], radar_data[6317:6803], radar_data[12636:13122], radar_data[16038:16524], radar_data[18468:18954]))
    radar_train_data = np.concatenate((radar_data[486:6317], radar_data[6803:12636], radar_data[13122:16038], radar_data[16524:18468], radar_data[18954:]))
    print(radar_test_data.shape, radar_train_data.shape)

    dis_label = np.concatenate((label1, label2))
    dis_label = dis_label[:,0,:]
    dis_test_label = np.concatenate((dis_label[:486], dis_label[6317:6803], dis_label[12636:13122], dis_label[16038:16524], dis_label[18468:18954]))
    dis_train_label = np.concatenate((dis_label[486:6317], dis_label[6803:12636], dis_label[13122:16038], dis_label[16524:18468], dis_label[18954:]))
    print(dis_test_label.shape, dis_train_label.shape)
    # print("radar_data_shape = " ,radar_data.shape, "label_data_shape = ", dis_label.shape)
    
    return radar_train_data, radar_test_data, dis_train_label, dis_test_label


def fully_connected_layer(input1, input_channel, output_channel, name = "fc", activate = "leaky", is_training=True):
    
    with tf.variable_scope(name):
        w = tf.get_variable("Wf", shape=[input_channel, output_channel], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("B", shape = [output_channel], initializer=tf.contrib.layers.xavier_initializer())
    if activate == "leaky":
        op = tf.nn.leaky_relu(tf.matmul(input1, w) + b)
    else:
        op = tf.matmul(input1, w) + b

    # op = tf.contrib.layers.batch_norm(op, is_training = is_training)

    return op

def conv1d_layer(input1, strides, filter_size, input_channel, number_filter, name = "conv1", is_training=True):
    
    with tf.variable_scope(name):
        w = tf.get_variable("W1d", shape = [filter_size, input_channel, number_filter], initializer = tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("B", shape = [number_filter], initializer= tf.contrib.layers.xavier_initializer())
    conv1 = tf.nn.conv1d(input1, w, stride = strides, padding="SAME")
    
    op = tf.nn.leaky_relu(conv1 + b)
    # op = tf.contrib.layers.batch_norm(op, is_training=is_training)
    return op

def L2_loss(denseOut, dis_label):

    L2_dis = tf.square(dis_label - denseOut)

    return tf.mean(L2_dis)

def run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label):

    tf.reset_default_graph()
    learning_rate = 0.01
    training_epoch = 3000
    batch_size = 10000
    log_dir = './summary_cnn_single_chirp_16bit_1/1dcov2'

    batch_test_size = 600
    batch_test_len = 0

    x_p = tf.placeholder(tf.float32, shape=[None, 256, 4], name= 'x_train')
    y_p = tf.placeholder(tf.float32, shape=[None, 3], name= 'y_train')

    conv1d = conv1d_layer(x_p, 1, 3, 4, 4, "conv1_layer1_en")
    conv1d = conv1d_layer(conv1d, 2, 3, 4, 8, "conv1_layer2_en")

    conv1d = conv1d_layer(conv1d, 1 , 3, 8, 8, "conv1_layer3_en")
    conv1d = conv1d_layer(conv1d, 2 , 3, 8, 16, "conv1_layer4_en")

    conv1d = conv1d_layer(conv1d, 1 , 3, 16, 16, "conv1_layer5_en")
    conv1d = conv1d_layer(conv1d, 2 , 3, 16, 32, "conv1_layer6_en")

    conv1d = conv1d_layer(conv1d, 1 , 3, 32, 32, "conv1_layer7_en")
    conv1d = conv1d_layer(conv1d, 2 , 3, 32, 64, "conv1_layer8_en")    

    flatten = tf.reshape(conv1d, [-1, 16*64 ])

    dense1 = fully_connected_layer(flatten, 16*64, 300, "fc1", "leaky")
    denseOut = fully_connected_layer(dense1, 300, 3, "fc2", "None")
    # denseOut= fully_connected_layer(dense2, 300, 3, "fcOut", "None")

    loss = L2_loss(denseOut, y_p)

    train_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_ops = train_optimiser.minimize(loss)

    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        
        sess.run(init)
        
        writer1 = tf.summary.FileWriter(log_dir+"/training")
        writer2 = tf.summary.FileWriter(log_dir+"/validation")



        for epoch in range(training_epoch):
            avg_loss = 0
            batch_len = 0
            total_batch = int(len(radar_train_data)/batch_size)
            
            for i in range(total_batch):
                x,c = sess.run([train_ops, loss], feed_dict= {x_p : radar_train_data[batch_len:(batch_len+batch_size)], y_p: dis_train_label[batch_len:(batch_len+batch_size)]})
                batch_len += batch_size
                avg_loss += c / total_batch


            training_acc = avg_loss
            test_acc = sess.run(loss, feed_dict = {x_p: radar_test_data[batch_test_len:(batch_test_len+batch_test_size)], y_p: dis_test_label[batch_test_len:(batch_test_len+batch_test_size)]})
            
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
            writer1.add_summary(summary_train, epoch)

            summary_test = tf.Summary()
            summary_test.value.add(tag = "ACC1", simple_value=test_acc)
            writer2.add_summary(summary_test, epoch)

        writer1.close()
        writer2.close()

def main():
    radar_train_data, radar_test_data, dis_train_label, dis_test_label = pre_process()
    run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label)

if __name__ == "__main__":
    main()