#%%
import numpy as np
import cmath as cm
import math as mp
# import scipy.fftpack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

def callBinfile(fname):
    # f_name = glob.glob('D:/NestData/10-5-2019-64-chirp-16bit/*.bin')
    # cut_frame = glob.glob('D:/NestData/9-24-2019-single-chirp-16bit/pos_process/frame*.txt')
    # IQarray_list = []
    # cut_frame_number = []
    # count = 0
    # print(f_name)

    # for name in cut_frame:
    #     f_cut = open(name,'r')
    #     for i in f_cut:
    #         cut_frame_number.append(int(i))
    # cut_frame_number = np.array(cut_frame_number)

    # for fname in f_name[:15]:

    print(fname)    
    fid = open(fname,'r')
    fid = np.fromfile(fid, np.int16)
    fid = fid[:65536000] # for 500 frame

    '''
        read data from bin file. Pls see data structure in MMwave radar device
        Raw data capture
        Data structure IQarray [number of RX, number of frame, chirp repeating, ADC resolution]
    '''
    frameNumber = 500
    totalChirpRepeat = 64
    numADCSamples = 256
    numADCbits = 16
    numRX = 4
    numLanes = 2
    fileSize = len(fid)
    # print(fileSize)
    IQarray = []
    numChirps = fileSize/2/numADCSamples/numRX
    fid1 = np.reshape(fid, (int(numChirps),-1))
    for i in range(0,fid1.shape[1],4):
        IQarray.append(fid1[:,i] + cm.sqrt(-1)*fid1[:,i+2])
        IQarray.append(fid1[:,i+1] + cm.sqrt(-1)*fid1[:,i+3])
    IQarray = np.array(IQarray)
    IQarray = np.reshape(IQarray,(numRX,numADCSamples,frameNumber,totalChirpRepeat))
    # IQarray = IQarray[:,:,14:,:]
    # print(cut_frame_number[count])
    # IQarray = np.swapaxes(IQarray, 0,2)
    # print(IQarray.shape)
    # IQarray_list.extend(IQarray)

    # count += 1

    # IQarray_list = np.array(IQarray_list)
    return np.swapaxes(np.swapaxes(IQarray, 1,2), 2,3)
    # return np.swapaxes(np.swapaxes(np.swapaxes(IQarray,0,2), 1,2), 2,3)


def animate(i):
    # line1.set_ydata(Comfft[0,i,15,:])
    print(i)


    # Re = IQarray[0,i,0,:].real
    # Im = IQarray[0,i,0,:].imag
    # line2.set_ydata(Re)
    # line3.set_ydata(Im)

    # frame = Comfft[0,i,:,:]
    # axIm.set_array(abs(frame))

    frame2 = Vefft[0,i,:,:]
    # frame2 = Vefft[3,i,:20,22:42]
    axDop.set_array(frame2)

    
    # return line1, line2, line3, axDop, axIm
    return axDop
    # return line1,

def runGraphInitial():
    global line1, line2, line3, axIm, axIm, axDop
    ''' 
        Initial condition fft 
        plot 1d fft
    '''
    fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # line1, = ax1.plot(Comfft[0,0,15,:])
    # line1, = ax1.plot(Comfft[0,0,:])
    '''
        plot IQ data (raw data)
    '''
    # ax2 = fig.add_subplot(222)
    # color_red = 'tab:red'
    # color_blue = 'tab:blue'
    # Re = IQarray[0,0,0,:].real
    # Im = IQarray[0,0,0,:].imag
    # line2, = ax2.plot(Re, color_red)
    # ax3 = ax2.twinx()
    # line3, = ax3.plot(Im, color_blue)

    '''
        imshow range-chirp , 
        imshow range-doppler 
    '''
    # axIm = fig.add_subplot(223)
    # frame = Comfft[0,0,:,:]
    # axIm = plt.imshow(abs(frame), aspect = 'auto', interpolation = 'catrom')
    # axIm = fig.add_subplot(223

    '''
        imshow range-doppler(velocity)
    # '''
    axDop = fig.add_subplot(224)
    veFrame = Vefft[0,0,:,:]
    # veFrame = Vefft[3,0,:20,22:42]
    axDop = plt.imshow(veFrame, aspect= 'auto', origin='lower' , cmap='jet'
    , extent=[-veFrame.shape[1]/2., veFrame.shape[1]/2., 0, veFrame.shape[0]-1], interpolation = 'catrom')
    plt.colorbar(axDop)

    ani = FuncAnimation(fig, animate, frames=470  ,interval = 100)
    plt.show()

def fftfunction():
    reIQarray = IQarray
    n = reIQarray.shape[3]
    Comfft = np.fft.fftshift(np.fft.fft(reIQarray) / n)
    # Comfft = abs(Comfft)
    # Comfft = 20*np.log10(Comfft/32767)
    # Comfft = Comfft[:,:,:,range(n//2)]
    # Comfft = Comfft[:,:,:,5:30]
    # print("dd", Comfft.shape)
    return Comfft

def fftVelocity():
    
    veFrame = Comfft[:,:,:,:]
    veFrame = np.swapaxes(veFrame,2,3)
    n = veFrame.shape[3]
    veFrame = np.fft.fftshift(np.fft.fft(veFrame)/n)
    # veFrame = abs(veFrame)
    # veFrame = 20*np.log10(veFrame/32767)
    # print(veFrame.shape[3]/2)

    # cut2 = veFrame[:, :, :, :int(veFrame.shape[3]/2)]
    # cut1 = veFrame[:, :, :, int(veFrame.shape[3]/2):]


    # frame2 = np.concatenate((cut1,cut2), axis=3)
    
    # plt.imshow(frame2[0,0,:,:], aspect = 'auto', origin='lower', cmap='jet')
    return veFrame
    # return frame2

#%%
def movingAvg():

    '''
    moving average background subtraction on dopppler image add window size 25
    '''

    veFrame = Vdop[:,:,:40,44:84]
    veFrame = np.swapaxes(veFrame, 0,1)
    print(veFrame.shape)
    veFrameSum = []
    # for i in range(veFrame.shape[0]-5): 
    #     veMeanFront = np.mean(veFrame[i:i+5,:,:,:], axis= 0)
    #     veMeanBack = np.mean(veFrame[i+6:i+11,:,:,:], axis= 0)
    #     veMeanAll = [veMeanFront, veMeanBack]
    #     veMeanAll = np.array(veMeanAll)
    #     veMeanAll = np.mean(veMeanAll, axis=0)
    #     veMeanSub = veFrame[i+5,:,:,:] - veMeanAll 
    #     veFrameSum.append(veMeanSub)
        

    # veFrameSum = [(j > 0)*j for j in veFrameSum]
    # veFrameSum = np.array(veFrameSum)
    # veFrameSum = abs(veFrameSum)
    # veFrameSum = np.swapaxes(veFrameSum,0,1)
    # print(veFrameSum.shape)
  
    # return veFrameSum

def movingAvg_OneD():
    
    '''
    moving averange background subtraction on 1-d fft 
    '''

    fftFrame = np.swapaxes(Comfft,0,1)
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

    # fftFrameSum = abs(fftFrameSum)
    fftFrameSum = np.swapaxes(fftFrameSum,0,1)
    print("swap second ", fftFrameSum.shape)

    return fftFrameSum

def conCatAllFrame():
    # signal_1 = np.load('D:/NestData/7-19-2019-new/signal_1.npy')
    # signal_2 = np.load('D:/NestData/7-19-2019-new/signal_2.npy')
    # signal_3 = np.load('D:/NestData/7-19-2019-new/signal_3.npy')
    # signal_4 = np.load('D:/NestData/7-19-2019-new/signal_4.npy')
    # signal_5_moving = np.load('D:/NestData/7-19-2019-new/signal_5_moving.npy')
    # signal_6_moving = np.load('D:/NestData/7-19-2019-new/signal_6_moving.npy')
    # signal_7 = np.load('D:/NestData/7-19-2019-new/signal_7.npy')
    # signal_8_moving = np.load('D:/NestData/7-19-2019-new/signal_8_moving.npy')
    # signal_9_moving = np.load('D:/NestData/7-19-2019-new/signal_9_moving.npy')
    # signal_10_test_moving = np.load('D:/NestData/7-19-2019-new/signal_10_test_moving.npy')
    # signal_11_test = np.load('D:/NestData/7-19-2019-new/signal_11_test.npy')

    # x_train = np.concatenate((signal_1,signal_2,signal_3,signal_4,signal_5,signal_6,signal_7,signal_8,signal_9), axis = 0)
    # x_test = np.concatenate((signal_10_test,signal_11_test), axis=0)


    # print(x_train.shape)
    # print(x_test.shape)

    x_train_2_moving = np.concatenate((signal_5_moving[:2400],signal_6_moving[:2400],signal_8_moving[:2400],signal_9_moving[:2400]), axis = 0)
    x_test_2_moving = signal_10_test_moving[:2400]

    print(x_train_2_moving.shape)
    print(x_test_2_moving.shape)

    np.save("D:/NestData/7-19-2019-new/x_train_2_moving",x_train_2_moving)
    np.save("D:/NestData/7-19-2019-new/x_test_2_moving",x_test_2_moving)

def main():
    global IQarray, Comfft, Vefft, Vdop
    name_count = 1
    Vefft_list = []
    f_name = glob.glob('D:/NestData/10-5-2019-64-chirp-16bit/*.bin')
    for fname in f_name:
        IQarray = callBinfile(fname)
        print(IQarray.shape)

        Comfft = fftfunction()
    # print("shape of comfft ", Comfft.shape)
    # plt.plot(20*np.log10(abs(Comfft[0,0,3,:])/32767))
    # plt.show()
        Comfft = movingAvg_OneD()
        Vefft = fftVelocity()
        Vefft_real = Vefft.real
        Vefft_imag = Vefft.imag
        Vefft = np.concatenate((Vefft_real,Vefft_imag))
        
        print(Vefft.shape)
        Vefft = np.swapaxes(Vefft, 0, 1)
        Vefft = Vefft[:,:,:20,22:42]
        print(Vefft.shape)
        Vefft_list.extend(Vefft)
        # Vefft = Vefft[:,:,:20,22:42]
        # print(Vefft.shape)
        # radar_name = "D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/radar_data_reduce/radar_data_reduce_" + str(name_count) + ".npy"
        # np.save(radar_name, Vefft)
        # print(radar_name)
        # name_count += 1
    # veFrame = Vefft[:,:,:20,22:42]
    # veFrame = np.swapaxes(veFrame,0,1)
    # veFrame = np.swapaxes(veFrame,1,2)
    # veFrame = np.swapaxes(veFrame,2,3)
        # Comfft = abs(Comfft)
        # Comfft = 20*np.log10(Comfft/32768)
    # print("shape before save =", veFrame.shape)
        # runGraphInitial()
    Vefft_list = np.array(Vefft_list)
    print(Vefft_list.shape)
    # Vefft_list = Vefft_list[:,:,:20,22:42]
    np.save("D:/NestData/10-5-2019-64-chirp-16bit/pos_process_new/radar_data_reduce/radar_data_reduce_all_moving_real_imag.npy",Vefft_list) 
    '''
    save file to npy
    '''
    # np.save("D:/NestData/7-19-2019-new/signal_10_test_moving", veFrame)
    # np.save("Comfft", Comfft)
    '''
    Load file
    '''
    # Vdop = np.load("doppler_img.npy")
    # Vefft = Vdop
    # print(Vdop.shape)
    # Vefft = movingAvg()
    # print(Vefft)
    # runGraphInitial()

    # Comfft = np.load("Comfft.npy")
    # Comfft = movingAvg_OneD()
    # Vefft = fftVelocity()
    # Comfft = abs(Comfft)
    # Comfft = 20*np.log10(Comfft/32768)
    # Comfft = Comfft[:,:,0,:]
    # print(Comfft.shape)
    # np.save("D:/NestData/9-24-2019-single-chirp-16bit/pos_process_new/radar_data", Comfft)
    # print(Comfft.shape)
    # runGraphInitial()
    # print(Comfft)
    
    '''
        concatinate file and save to x_train,test
    '''
    # conCatAllFrame()
    
if __name__ == '__main__':
    main()  




#%%
