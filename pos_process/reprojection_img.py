import cv2
import numpy as np
import glob
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from time import sleep

dis_test_label = np.load("C:/Users/nakorn-vision/Documents/PythonFile/NestProject/Nest_Model/pos_process/test_data/dis_test_data_complex_exp_reduce_test(train).npy")
dis_predict = np.load('C:/Users/nakorn-vision/Documents/PythonFile/NestProject/Nest_Model/pos_process/prediction_data/predict_data_complex_exp_reduce_test(train).npy')
# dis_label = np.concatenate((label1, label2))
print(dis_test_label.shape, dis_predict.shape)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

colors = [1.0,0,0,0.5]

dis_test_label = dis_test_label/100
dis_predict = dis_predict/100


sp0 = gl.GLScatterPlotItem(pos=dis_test_label[:], color=colors)
w.addItem(sp0)

sp1 = gl.GLScatterPlotItem(pos=dis_predict[:])
w.addItem(sp1)

# for i in range(500):

# i = 0
# def update():
    
#     global i 

#     sp2 = gl.GLScatterPlotItem(pos=dis_test_label[i], color=colors )
#     w.addItem(sp2)


#     sp3 = gl.GLScatterPlotItem(pos=dis_predict[i])
#     w.addItem(sp3)

#     i += 1

# time = QtCore.QTimer()
# time.timeout.connect(update)
# time.start(0.1)


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

