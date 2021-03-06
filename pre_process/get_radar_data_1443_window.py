#%%
import numpy as np
import cmath as cm
import math as mp
# import scipy.fftpack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import re
from sklearn.neighbors import NearestNeighbors

def callBinfile(fname):
    
    # print(fname)    
    fid = open(fname,'r')
    fid = np.fromfile(fid, np.int16)
    fid = fid[:393216000] # for 500 frame

    '''
        read data from bin file. Pls see data structure in MMwave radar device
    '''
    frameNumber = 500
    numADCSamples = 1024
    numADCbits = 16
    numTx = 3
    numChirps = 32
    numLanes = 4
    '''
    --------------------------------------------------------------------------------
    '''
    print(fid.shape)
    Tx1, Tx2, Tx3 = list(), list(), list()
    adcData = np.reshape(fid, (-1,numLanes*2))
    adcData_complex = adcData[:,[0,1,2,3]] + cm.sqrt(-1)*adcData[:,[4,5,6,7]]
    adcData_complex = np.reshape(adcData_complex,(frameNumber,numADCSamples*numTx*numChirps,-1))
    for i in range(0, numTx*numChirps*numADCSamples, numTx*numADCSamples):
        Tx1.append(adcData_complex[:,i:i+numADCSamples,:])
        Tx2.append(adcData_complex[:,i+numADCSamples:i+(numADCSamples*2),:])
        Tx3.append(adcData_complex[:,i+(numADCSamples*2):i+(numADCSamples*3),:])
    adcData_Tx1 = np.array(Tx1)
    adcData_Tx2 = np.array(Tx2)
    adcData_Tx3 = np.array(Tx3)
 
    # print(adcData_Tx1.shape)
    # print(adcData_Tx2.shape)
    # print(adcData_Tx3.shape)
    list_adc = [adcData_Tx1, adcData_Tx2, adcData_Tx3]
    # list_adc = [adcData_Tx1]
    adcData_Tx_all = np.concatenate(list_adc, axis = 3)
    print(adcData_Tx_all.shape)

    # plt.plot(adcData_Tx1[0,1,:,0])
    # plt.plot(adcData_Tx2[0,1,:,0])
    # plt.plot(adcData_Tx3[0,1,:,0])
    # plt.plot(adcData_Tx_all[0,1,:,0])
    # plt.plot(adcData_Tx_all[32,1,:,0])
    # plt.plot(adcData_Tx_all[64,1,:,0])

    # plt.show()
    
    '''
        return Tx1, Tx2, Tx3 - TDM - MIMO
        with shape (chirp, frame, adc_sample, Rx)
    '''
    return np.swapaxes(adcData_Tx_all, 0,1) 
    
def fileChecking():
    
    count_all = 0
    f_name = glob.glob('D:/NestData/3tx-32chirp-jaco-55times_all/3tx*')
    _marker = object()
    obj_uncut = []

    for fname in f_name :
        print(fname)
        ch = 0
        count_index = 0
        obj = [None]
        kname = fname + "/pos_process_label_2/radar_pos_label.npy"
        dname = fname + "/pos_process_label_2/frame_number.txt"
        print(kname)
        test = np.load(kname)
        check_list = [None]*test.shape[0]
        
        with open(dname) as f:
            lines = [line.rstrip() for line in f]
            lines = np.array(lines)
            lines = lines.astype(np.int)    
            print(lines.shape)

        for line in lines:
            # print(line)
            count_index += 1
            for i in range(ch,ch+line,1):
                # print(ch)
                if line < 481:
                    check_list[i] = _marker
                else:
                    check_list[i] = test[i]
            if line < 481:
                print(count_index)
            
            ch += line
        obj[:] = [v for v in check_list if v is not _marker]
        obj = np.array(obj)
        for blank in range(0,obj.shape[0],481):
            count_all += 1
            subf = 'D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label/position_' + str(count_all) 
            np.save(subf, obj[blank:blank+481])

        print(np.array(obj).shape)
        # obj_uncut.extend(obj)
        # print(np.array(obj_uncut).shape)
        # print(np.array(obj)[:])
        

    # obj_uncut = np.array(obj_uncut)
    # print("ss" , obj_uncut[472])

    # print(np.where(obj_uncut == 0)[0])
    # count = []
    # print(obj_uncut.shape)
    # for i in range(obj_uncut.shape[0]):
    #     if obj_uncut[i,0,0] == 0:
    #         count.append(i)
    # obj_cut = np.delete(obj_uncut, count, axis=0)
    # print(np.where(obj_uncut == 0)[0])
    # print(obj_cut.shape)
    # np.save('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_2/radar_pos_label_deleted.npy',np.array(obj_cut))
    # np.save('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_2/deleted_index.npy',np.array(count))
    # print(np.array(count))

def animate(i):

    # line1.set_ydata(range_fft[i,5,:,3])
    # line1.set_ydata(range_fft[i,:])
    
    Re = IQall[i,0,:,0].real
    Im = IQall[i,0,:,0].imag
    line2.set_ydata(Re)
    line3.set_ydata(Im)
    print(IQall[i])

    # frame = range_fft[i,:,:,3]
    # axIm.set_array(abs(frame))

    # frame2 = range_doppler[i,:,:,3]
    # frame2 = Vefft[0,i,:20,22:42]
    # axDop.set_array(frame2)

    
    return line2, line3
    # return line1, line2, line3, axDop, axIm

def runGraphInitial():
    
    global line1, line2, line3, axIm, axIm, axDop

    fig = plt.figure(1)
    ''' 
        Initial condition fft 
        plot 1d fft
    '''
    # ax1 = fig.add_subplot(221)
    # line1, = ax1.plot(range_fft[0,5,:,3])
    # line1, = ax1.plot(range_fft[0,:])
    # ax1.set_ylim([-100,-10])

    '''
        plot IQ data (raw data)
    '''
    ax2 = fig.add_subplot(222)
    color_red = 'tab:red'
    color_blue = 'tab:blue'
    Re = IQall[0,0,:,0].real
    Im = IQall[0,0,:,0].imag
    line2, = ax2.plot(Re, color_blue)
    ax3 = ax2.twinx()
    line3, = ax3.plot(Im, color_blue)

    '''
        imshow range-chirp , 
        imshow range-doppler 
    # '''
    # axIm = fig.add_subplot(223)
    # frame = range_fft[0,:,:,3]
    # axIm = plt.imshow(abs(frame), aspect = 'auto', interpolation = 'catrom')

    '''
        imshow range-doppler(velocity)
    # '''
    # axDop = fig.add_subplot(224)
    # veFrame = range_doppler[0,:,:,3]
    # veFrame = Vefft[0,0,:20,22:42]
    
    # axDop = plt.imshow(veFrame, aspect= 'auto', origin='lower' , cmap='jet'
    # , extent=[-veFrame.shape[1]/2., veFrame.shape[1]/2., 0, veFrame.shape[0]-1], interpolation = 'catrom')
    # axDop = plt.imshow(veFrame, aspect= 'auto', origin='lower' , cmap='jet')
    # plt.colorbar(axDop)

    ani = FuncAnimation(fig, animate, frames=500  ,interval = 100)
    plt.show()

def fft_range_function():
    n  = IQall.shape[2]
    range_fft = np.fft.fft(IQall, axis=2) / n
    # range_fft = abs(range_fft)
    # range_fft = 20*np.log10(range_fft/32767)
    
    return range_fft

def fftVelocity():
    
    n = range_fft.shape[1]
    veFrame = np.fft.fftshift(np.fft.fft(range_fft, axis=1) / n, axes=1)
    # veFrame = abs(veFrame)

    return abs(veFrame) ## for plot
    # return veFrame

def movingAvg_OneD():
    
    '''
    moving averange background subtraction on 1-d fft 
    '''

    fftFrame = range_fft
    print("swap first ", fftFrame.shape)
    fftFrameSum = []
    for i in range(fftFrame.shape[0]-14):

        fftMeanFront = np.mean(fftFrame[i:i+14,:,:,:], axis=0)
        # print(fftMeanFront)
        # fftMeanBack = np.mean(fftFrame[i+16:i+21,:,:,:], axis=0)
        # fftMeanAll = [fftMeanFront, fftMeanBack] 
        # fftMeanAll = np.array(fftMeanAll)
        # fftMeanAll = np.mean(fftMeanAll, axis=0)
        fftMeanSub = fftFrame[i+14,:,:,:] - fftMeanFront
        fftFrameSum.append(fftMeanSub)

    fftFrameSum = np.array(fftFrameSum)
    '''
    '''

    # # fftFrameSum = abs(fftFrameSum)
    # fftFrameSum = np.swapaxes(fftFrameSum,0,1)
    # print("swap second ", fftFrameSum.shape)

    return fftFrameSum

def background_average():
    range_fft_avg = np.mean(range_fft, axis=0)
    np.save('D:/NestData/SNR_data2/radar_5/bg_5_avg.npy', range_fft_avg)
    print(range_fft_avg.shape)

def label_knn(file):
    global file_number
    
    dis_deleted = []
    deleted_index = []
    dis_train = np.load(file)
    # print(dis_train)
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
    # print(dis_deleted)
    # dis_train = np.delete(dis_train, deleted_index, axis=0)
    # print(dis_train.shape)
    # print(dis_train[400])
    # subf = 'D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label/label_processed_' + str(file_number)
    # subd = 'D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label/label_deleted_index_' + str(file_number)
    # np.save(subf, dis_train)
    # np.save(subd, deleted_index)
    print(file_number)
    file_number += 1

    return dis_train, deleted_index
    


def main():

    global IQall, range_fft, file_number, range_doppler

    name_count = 0
    file_number = 1
    range_doppler_list = []
    range_fft_list = []
    signal_modulation_am = []

    folder_name = glob.glob('D:/NestData/3tx-32chirp-jaco-55times_all/3tx-32chirp-jaco-55times_pos*')
    folder_label_name = glob.glob('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage_3/label_no_process/*')
    folder_label_name.sort(key=lambda f: int(re.sub('\D','',f)))
    folder_name.sort(key=lambda f: int(re.sub('\D','',f)))



    for sub_f in folder_name:
        # print(sub_f)
        bin_insub = sub_f + '/*.bin'
        sub_bin = glob.glob(bin_insub)
        sub_bin.sort(key=lambda f: int(re.sub('\D','',f)))

        for ss in sub_bin:
            count_zero_index = 0
            radar_zero = []
            print("radar =", ss,"----------- radar =", folder_label_name[name_count])
            dis_train, deleted_index = label_knn(folder_label_name[name_count])
            IQall = callBinfile(ss)


            
            for iq in IQall:
                check_zero = np.count_nonzero(iq)
                if check_zero < 380000:
                    radar_zero.append(count_zero_index)
                count_zero_index += 1 
            radar_zero = np.array(radar_zero)
            radar_label_zero_index = np.concatenate((radar_zero, deleted_index))
            _, ix = np.unique(radar_label_zero_index, return_index=True)
            print(radar_zero, deleted_index)
            print(radar_label_zero_index[np.sort(ix)])
            plt.imshow(np.float32(IQall[89,:,:,0]))
            plt.show()
            name_count += 1
             
            # print(IQall)
            # range_fft = fft_range_function()
            # range_fft = range_fft[:,:,:32,:]
            # print(range_fft.shape, "check", range_fft[0,0,0,0])
            # range_fft_real = np.float16(range_fft.real)
            # range_fft_imag = np.float16(range_fft.imag)
            # range_fft_all = np.concatenate((range_fft_real, range_fft_imag), axis= 3)
            # print(range_fft_all.shape, "check", range_fft_all[0,0,0,0], range_fft_all[0,0,0,12])
            # range_fft = abs(range_fft)
            # background_average()
            # signal_modulation_am.append(range_fft) 
            # range_fft = movingAvg_OneD()
            # range_doppler = fftVelocity()
            # # range_doppler = range_doppler[:,:,:32,:]
            # range_doppler_real = np.float16(range_doppler.real)
            # range_doppler_imag = np.float16(range_doppler.imag)
            # range_doppler_all = np.concatenate((range_doppler_real, range_doppler_imag), axis= 3)
            # print(range_doppler_all.shape)
            # print("check", range_doppler_real[0,0,0,0], range_doppler_imag[0,0,0,0])
            # range_doppler_list.extend(range_doppler_all)
            # range_fft_list.extend(range_fft_all)
            # # print(range_doppler_all[0,0,:,10])
            # # print(range_doppler_real[0,0,:,0])
            # # print(range_doppler_imag[0,0,:,0])
            # # print(range_doppler_real[0,0,0,0], range_doppler_real_r[0,0,0,0])
            # print(np.array(range_doppler_list).shape)
            # runGraphInitial()
        # print(name_count)
    
    # print(name_count)
    # range_doppler_list = np.array(range_doppler_list)
    # range_fft_list = np.array(range_fft_list)
    # print(range_doppler_list.shape)
    # print(range_fft_list)
    # file_name = "D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/radar_fft_data_all_real_imag_.npy"
    # np.save(file_name, range_fft_list) 
    # name_count += 1

    '''
    save file to npy
    '''
    # np.save("D:/NestData/7-19-2019-new/signal_10_test_moving", veFrame)
    # np.save("Comfft", Comfft)
    
    
if __name__ == '__main__':
    main()
    
    # fileChecking()  
    # test = np.load('D:/NestData/3tx-32chirp-jaco-55times_all/data_usage/radar_data_all_real_imag_.npy')
    # print(test[0,0,:,10])


# %%
