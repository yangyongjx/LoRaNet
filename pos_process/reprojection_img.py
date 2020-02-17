import cv2
import numpy as np
import glob
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from time import sleep

dis_test_label = np.load("D:/PythonFile/NestProject/Nest_Model/pos_process/test_data/dis_test_data_complex_exp_shuffle_3.npy")
dis_test_predict = np.load('D:/PythonFile/NestProject/Nest_Model/pos_process/prediction_data/predict_data_complex_exp_shuffle_3.npy')

dis_train_label = np.load("D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/radar_pos_label_deleted.npy")
dis_train_predict = np.load('D:/PythonFile/NestProject/Nest_Model/pos_process/prediction_data/predict_data_complex_exp_shuffle_4(train).npy')

# dis_label = np.concatenate((label1, label2))
print(dis_train_label.shape, dis_test_predict.shape)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

colors_1 = [1.0,0,0,0.5]
colors_2 = [0,1.0,0,0.5]

# dis_test_predict = dis_test_predict[:,0,:]

dis_test_label = dis_test_label/100
dis_test_predict = dis_test_predict/100
dis_train_label = dis_train_label/100
dis_train_predict = dis_train_predict/100
dis_train_label = dis_train_label[:500]


# dis_test_label = dis_test_label[:,1:3]
# dis_predict = dis_predict[:,1:3]
# print(dis_test_label.shape)

# sp0 = gl.GLScatterPlotItem(pos=dis_test_label[:], color=colors_1)
# w.addItem(sp0)

# sp1 = gl.GLScatterPlotItem(pos=dis_test_predict[:])
# w.addItem(sp1)

sp2 = gl.GLScatterPlotItem(pos=dis_train_label[:], color=colors_2)
w.addItem(sp2)

# sp3 = gl.GLScatterPlotItem(pos=dis_train_predict[:])
# w.addItem(sp3)


# i = 0
# def update():
    
#     global i 

#     sp2 = gl.GLScatterPlotItem(pos=dis_train_label[i], color=colors_1)
#     w.addItem(sp2)


#     # sp3 = gl.GLScatterPlotItem(pos=dis_test_predict[i])
#     # w.addItem(sp3)
#     print(i)
#     i += 1

# time = QtCore.QTimer()
# time.timeout.connect(update)
# time.start(5)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

