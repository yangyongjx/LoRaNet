#%%

import glob
import numpy as np
# images = glob.glob('D:/NestData/9-24-2019-single-chirp-16bit/*.bin')
# frame = glob.glob('D:/NestData/9-24-2019-single-chirp-16bit/pos_process/frame*.txt')
# # pos = np.load('D:/NestData/9-24-2019-single-chirp-16bit/radar_pos_label.npy')
# # print(pos.shape)
# for fname in images:
#     print(fname)

# for name in frame:
#     f = open(name,'r')
#     for i in f:
#         print(int(i))

#%%
test = np.load('D:/NestData/3tx-32chirp-jaco-55times_all/3tx-32chirp-jaco-55times_pos1/pos_process_label/radar_pos_label.npy')
# test = test[:11664]
print(np.where(test == 0)[0])
count = []
print(test.shape)
# print(test[0])
for i in range(test.shape[0]):
    if test[i,0,0] == 0:
        count.append(i)
test = np.delete(test, count, axis=0)
print(np.where(test == 0)[0])
print(test.shape)
# np.save('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/radar_pos_label_deleted.npy',test)
# np.save('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/deleted_index.npy',np.array(count))
print(np.array(count).shape)


