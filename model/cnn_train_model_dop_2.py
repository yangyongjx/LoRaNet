import tensorflow as tf
import os
# import cv2
import numpy as np
import re
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
from tensorflow.compat.v1 import ConfigProto
import collections
from sklearn.model_selection import train_test_split
import tensorflow.python.framework.dtypes
sss
os.environ["CUDA_VISIBLE_DEVICES"]="2"

def pre_process():
    # dis_label = np.load("/home/nakorn/data/radar_pos_label_deleted.npy")
    # dis_label = np.load("D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/radar_pos_label_deleted.npy")
    # deleted = np.load("D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/deleted_index.npy")   
    # print(dis_label.shape)
    # deleted = np.load("/home/nakorn/data/deleted_index.npy")   
    # dis_label = dis_label[:,0,:]
    # radar_data = 1
    radar_data = np.load("D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/radar_data_all_real_imag_new_cut.npy")
    
    deleted = np.load("D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/deleted_index.npy")
    dis_label = np.load("D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/radar_pos_label_deleted.npy")

    

    print(radar_data.shape, dis_label.shape)
    
    radar_data = np.delete(radar_data, deleted, axis=0)

    dis_label = dis_label[:,0,:]
    
    radar_data = radar_data[:5000]
    dis_label = dis_label[:5000]
    
    print(radar_data.shape, dis_label.shape)
   
    radar_data = np.float32(radar_data)
    dis_label = np.float32(dis_label)

   

    radar_train_data, radar_test_data, dis_train_label, dis_test_label = train_test_split(radar_data, dis_label, test_size=0.3)
        
    
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
    batch_train_size = 500
    batch_test_size = 500
    batch_test_len = 0

    log_dir = './summary_raw_data/summarize_jaco_data_test/2dcov2'
    model_path = './saved_model/model_jaco_data_test/4plot.ckpt'
    # model_load = './saved_model/model_cnn_6_layers_complex_1_2d/4plot.ckpt'

    training_phase = tf.placeholder(tf.bool, name = 'training_phase')
    print(type(radar_train_data[0,0,0,0]), type(dis_test_label[0,0]))
    train_data, train_label = tf.compat.v1.train.shuffle_batch([radar_train_data, dis_train_label], enqueue_many=True, batch_size=batch_train_size, 
        capacity=int(batch_train_size*2), min_after_dequeue=int(batch_train_size/3), allow_smaller_final_batch = True)
    test_data, test_label = tf.compat.v1.train.batch([radar_test_data, dis_test_label], enqueue_many=True, batch_size=batch_test_size, 
        capacity=batch_test_size, allow_smaller_final_batch=True)
    print("ddd")
    x_p, y_p =  tf.cond(training_phase, lambda: (train_data, train_label) , lambda: (test_data, test_label))
    # x_p = tf.placeholder(tf.float32, shape=[None, 20, 20, 8], name= 'x_train')
    # y_p = tf.placeholder(tf.float32, shape=[None, 3], name= 'y_train')
    x_p = tf.cast(x_p, dtype=tf.float32)
    y_p = tf.cast(y_p, dtype=tf.float32)

    print(x_p)
    conv2d = conv2d_layer(x_p, 1, 3, 24, 8, "conv2_layer1_en")
    conv2d = conv2d_layer(conv2d, 2, 3, 8, 16, "conv2_layer2_en")
    # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training = training_phase) 

    conv2d = conv2d_layer(conv2d, 1 , 3, 16, 16, "conv2_layer3_en")
    conv2d = conv2d_layer(conv2d, 2 , 3, 16, 32, "conv2_layer4_en")
    # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training = training_phase)

    conv2d = conv2d_layer(conv2d, 1 , 3, 32, 32, "conv2_layer5_en")
    conv2d = conv2d_layer(conv2d, 2 , 3, 32, 64, "conv2_layer6_en")
    # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training = training_phase)

    # conv2d = conv2d_layer(conv2d, 1 , 3, 32, 32, "conv1_layer7_en")
    # conv2d = conv2d_layer(conv2d, 2 , 3, 32, 64, "conv1_layer8_en")    

    flatten = tf.reshape(conv2d, [-1, 4*4*64 ])

    dense1, none = fully_connected_layer(flatten, 4*4*64, 200, "fc1")
    # dense1 = tf.contrib.layers.batch_norm(dense1, is_training = training_phase)
    none, denseOut  = fully_connected_layer(dense1, 200, 3, "fc2")
    # denseOut= fully_connected_layer(dense2, 200, 3, "fcOut", "None")
    loss, hist = L2_loss(denseOut, y_p)


    learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, 5000, 0.96, staircase=True)

    train_optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    # update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # with tf.control_dependencies(update_ops):
    train_ops = train_optimiser.minimize(loss, global_step=global_step)

    hist_ops = tf.summary.histogram("histrogram_l2", hist)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    
        
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
    print(radar_train_data.shape, radar_test_data.shape, dis_train_label.shape, dis_test_label.shape)
    run_graph(radar_train_data, radar_test_data, dis_train_label, dis_test_label)
   


if __name__ == "__main__":
    main()