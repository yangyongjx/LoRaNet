#%%
import glob
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors

f_name = glob.glob('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label_no_process/*')
f_name.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(f_name)
file_number = 1
for file in f_name[99:100]:
    print(file)
    dis_deleted = []
    deleted_index = []
    dis_train = np.load(file)
    print(dis_train)
    dis_train = dis_train[:,0,:]
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(dis_train)
    distances, indices = nbrs.kneighbors(dis_train)
    
    count = 0
    for dis in distances:
        if  dis[dis>40].any():
            # print(count)s
            dis_deleted.append(dis)
            deleted_index.append(count)
        count += 1
    dis_deleted = np.array(dis_deleted)
    deleted_index = np.array(deleted_index)
    print(dis_deleted, deleted_index)
    dis_train = np.delete(dis_train, deleted_index, axis=0)
    print(dis_train.shape)
    print(dis_train[400])
    # subf = 'D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label/label_processed_' + str(file_number)
    # subd = 'D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label/label_deleted_index_' + str(file_number)
    # np.save(subf, dis_train)
    # np.save(subd, deleted_index)
    file_number += 1


  
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
# distances, indices = nbrs.kneighbors(X)
# print(indices)
# print(distances)


#%%
# test = np.load('D:/NestData/3tx-32chirp-jaco-55times_all/3tx-32chirp-jaco-55times_pos1/pos_process_label/radar_pos_label.npy')
# # test = test[:11664]
# print(np.where(test == 0)[0])
# count = []
# print(test.shape)
# # print(test[0])
# for i in range(test.shape[0]):
#     if test[i,0,0] == 0:
#         count.append(i)
# test = np.delete(test, count, axis=0)
# print(np.where(test == 0)[0])
# print(test.shape)
# # np.save('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/radar_pos_label_deleted.npy',test)
# # np.save('D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/deleted_index.npy',np.array(count))
# print(np.array(count).shape)

# -------------- cut data radar 500 to 481 function -------------------------

# data = np.load('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_2/radar_fft_data_all_real_imag_.npy')
# data_new = []
# for i in range(0, data.shape[0], 500):
#     data_new.extend(data[i+19:i+500])

# print(np.array(data_new).shape)
# data_new = np.array(data_new)
# np.save('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_2/radar_fft_data_all_real_imag_cut', data_new)
