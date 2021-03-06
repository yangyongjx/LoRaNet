import tensorflow as tf
import os
import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.compat.v1 import ConfigProto
import collections

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

    radar_test_data = np.concatenate((radar_data[:486], radar_data[6317:6803], radar_data[12636:13122], radar_data[16038:16524], radar_data[18468:18954]))
    # radar_test_data = np.concatenate((radar_data[6317:6803], radar_data[12636:13122], radar_data[16038:16524], radar_data[18468:18954]))
    radar_train_data = np.concatenate((radar_data[486:6317], radar_data[6803:12636], radar_data[13122:16038], radar_data[16524:18468], radar_data[18954:]))
    # radar_train_data = np.concatenate((radar_data[:6317], radar_data[6803:12636], radar_data[13122:16038], radar_data[16524:18468], radar_data[18954:]))
    print(radar_test_data.shape, radar_train_data.shape)

    dis_label = np.concatenate((label1, label2))
    dis_label = dis_label[:,0,:]
    # print(dis_label.shape)

    dis_test_label = np.concatenate((dis_label[:486], dis_label[6317:6803], dis_label[12636:13122], dis_label[16038:16524], dis_label[18468:18954]))
    # dis_test_label = np.concatenate((dis_label[6317:6803], dis_label[12636:13122], dis_label[16038:16524], dis_label[18468:18954]))
    dis_train_label = np.concatenate((dis_label[486:6317], dis_label[6803:12636], dis_label[13122:16038], dis_label[16524:18468], dis_label[18954:]))
    # dis_train_label = np.concatenate((dis_label[:6317], dis_label[6803:12636], dis_label[13122:16038], dis_label[16524:18468], dis_label[18954:]))
    print(dis_test_label.shape, dis_train_label.shape)
    # print("radar_data_shape = " ,radar_data.shape, "label_data_shape = ", dis_label.shape)
    
    return radar_train_data, radar_test_data, dis_train_label, dis_test_label


def fully_connected_layer(input1, input_channel, output_channel, name = "fc"):
    
    with tf.variable_scope(name):
        w = tf.get_variable("Wf", shape=[input_channel, output_channel], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("B", shape = [output_channel], initializer=tf.contrib.layers.xavier_initializer())

    return tf.nn.leaky_relu(tf.matmul(input1, w) + b), (tf.matmul(input1, w) + b)

def conv2d_layer(input1, stride_nn, filter_size, input_channel, number_filter, name = "conv2" ):
    
    with tf.variable_scope(name):
        w = tf.get_variable("W2d", shape = [filter_size, filter_size, input_channel, number_filter], initializer= tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("B", shape = [number_filter], initializer = tf.contrib.layers.xavier_initializer())    
    conv2 = tf.nn.conv2d(input1, w, strides = [1, stride_nn, stride_nn, 1], padding="SAME")
   
    return tf.nn.leaky_relu(conv2 + b)

def L2_loss(denseOut, dis_label):

    L2_dis = tf.square(dis_label - denseOut)

    return tf.sqrt(tf.reduce_mean(L2_dis)), L2_dis

def run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label):

    tf.reset_default_graph()
 
    # learning_rate = 0.01
    test_acc_prv = collections.deque(maxlen=2)
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01
    
    training_epoch = 10000
    batch_train_size = 1500
    batch_test_size = 2430
    batch_test_len = 0

    log_dir = './summary_raw_data/summary_cnn_6_layers_complex_shuffle_4/2dcov2'
    model_path = './saved_model/model_cnn_6_layers_complex_shuffle_4/4plot.ckpt'
    # model_load = './saved_model/model_cnn_6_layers_complex_1_2d/4plot.ckpt'

    training_phase = tf.placeholder(tf.bool, name = 'training_phase')

    train_data, train_label = tf.train.shuffle_batch([radar_train_data, dis_train_label], enqueue_many=True, batch_size=batch_train_size, 
        capacity=3000, min_after_dequeue=500, allow_smaller_final_batch = True)
    test_data, test_label = tf.train.batch([radar_test_data, dis_test_label], enqueue_many=True, batch_size=batch_test_size, 
        capacity=batch_test_size, allow_smaller_final_batch=True)

    x_p, y_p =  tf.cond(training_phase, lambda: (train_data, train_label) , lambda: (test_data, test_label))
    # x_p = tf.placeholder(tf.float32, shape=[None, 20, 20, 8], name= 'x_train')
    # y_p = tf.placeholder(tf.float32, shape=[None, 3], name= 'y_train')
    x_p = tf.cast(x_p, dtype=tf.float32)
    y_p = tf.cast(y_p, dtype=tf.float32)

    print(x_p)
    conv2d = conv2d_layer(x_p, 1, 3, 8, 4, "conv2_layer1_en")
    conv2d = conv2d_layer(conv2d, 2, 3, 4, 4, "conv2_layer2_en")
    # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training = training_phase) 

    conv2d = conv2d_layer(conv2d, 1 , 3, 4, 8, "conv2_layer3_en")
    conv2d = conv2d_layer(conv2d, 2 , 3, 8, 8, "conv2_layer4_en")
    # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training = training_phase)

    conv2d = conv2d_layer(conv2d, 1 , 3, 8, 16, "conv2_layer5_en")
    conv2d = conv2d_layer(conv2d, 2 , 3, 16, 16, "conv2_layer6_en")
    # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training = training_phase)

    # conv1d = conv2d_layer(conv1d, 1 , 3, 32, 32, "conv1_layer7_en", "leaky")
    # conv1d = conv2d_layer(conv1d, 2 , 3, 32, 64, "conv1_layer8_en", "leaky")    

    flatten = tf.reshape(conv2d, [-1, 4*4*16 ])

    dense1, none = fully_connected_layer(flatten, 4*4*16, 200, "fc1")
    # dense1 = tf.contrib.layers.batch_norm(dense1, is_training = training_phase)
    none, denseOut  = fully_connected_layer(dense1, 200, 3, "fc2")
    # denseOut= fully_connected_layer(dense2, 200, 3, "fcOut", "None")
    loss, hist = L2_loss(denseOut, y_p)


    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 1200, 0.96, staircase=True)

    train_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # with tf.control_dependencies(update_ops):
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
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        '''
        restore model path
        '''
        # saver.restore(sess, model_load)
        # print("model restore from file: %s" % model_load)


        for epoch in range(training_epoch):
            avg_loss = 0
            batch_len = 0
            total_batch = int(len(radar_train_data)/batch_train_size)
            
            for i in range(total_batch):
                x,c  = sess.run([train_ops, loss], feed_dict={training_phase: True})
                avg_loss += c / total_batch


            training_acc = avg_loss
            test_acc, histrogram, lr = sess.run([loss, hist_ops, learning_rate], feed_dict = {training_phase: False})
            
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
            if (epoch+1)%50 == 0 :
                test_acc_prv.append(test_acc)
                print(test_acc_prv)
                if np.array(test_acc) <= np.array(test_acc_prv)[0]:
                    print("-----Save model----- epoch + 1")
                    print("Last_test_acc =", np.array(test_acc_prv)[0], "Current_test_acc", np.array(test_acc))
                    save_path = saver.save(sess, model_path)
                    print("Model saved in file: %s" %save_path)
        
        coord.request_stop()
        coord.join(threads)

    writer1.close()
    writer2.close()

def main():
    radar_train_data, radar_test_data, dis_train_label, dis_test_label = pre_process()
    # plt.imshow(radar_train_data[0,:,:,0])
    # plt.show()
    radar_train_data = radar_train_data[:3800]
    dis_train_label = dis_train_label[:3800]
    radar_test_data = radar_test_data[:600]
    dis_test_label = dis_test_label[:600]
    print("shape new", radar_train_data.shape, radar_test_data.shape, dis_train_label.shape, dis_test_label.shape)

    # run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label)

if __name__ == "__main__":
    main()