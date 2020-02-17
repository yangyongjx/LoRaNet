#%%
import numpy as np
import cmath as cm
import math as mp
# import scipy.fftpack
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob

def callBinfile(fname):
    
    print(fname)    
    fid = open(fname,'r')
    fid = np.fromfile(fid, np.int16)
    # fid = fid[:65536000] # for 500 frame

    '''
        read data from bin file. Pls see data structure in MMwave radar device
        Raw data capture
        Data structure IQarray [number of RX, number of frame, chirp repeating, ADC resolution]
    '''
    frameNumber = 500
    totalChirpRepeat = 64
    numADCSamples = 256
    numADCbits = 16
    numRX = 12
    numLanes = 2

    fileSize = len(fid)
    print(fileSize)
    IQarray = []
    numChirps = fileSize/2/numADCSamples/numRX
    print(numChirps)
    fid1 = np.reshape(fid, (int(numChirps),-1))
    for i in range(0,fid1.shape[1],4):
        IQarray.append(fid1[:,i] + cm.sqrt(-1)*fid1[:,i+2])
        IQarray.append(fid1[:,i+1] + cm.sqrt(-1)*fid1[:,i+3])
    IQarray = np.array(IQarray)
    IQarray = np.reshape(IQarray,(numRX,numADCSamples,frameNumber,totalChirpRepeat))
    
    return np.swapaxes(np.swapaxes(IQarray, 1,2), 2,3)
 


def animate(i):
    line1.set_ydata(abs(Comfft[0,i,32,120:]))
    print(i)
    Re = IQarray[0,i,0,:].real
    Im = IQarray[0,i,0,:].imag
    line2.set_ydata(Re)
    line3.set_ydata(Im)

    frame = Comfft[0,i,:,120:]
    axIm.set_array(abs(frame))

    # frame2 = Vefft[0,i,:,:]
    frame2 = Vefft[0,i,:20,22:42]
    axDop.set_array(frame2)

    
    # return line1, line2, line3, axDop, axIm
    return axDop, line1, line2, line3, axIm
    # return line2, line3, axDop

def runGraphInitial():
    global line1, line2, line3, axIm, axIm, axDop
    ''' 
        Initial condition fft 
        plot 1d fft
    '''
    fig = plt.figure(1)
    # ax1 = fig.add_subplot(221)
    # ax1.set_ylim(0,10)
    # line1, = ax1.plot(Comfft[0,0,0,:])
    
    # line1, = ax1.plot(Comfft[0,0,:])
    '''
        plot IQ data (raw data)
    '''
    ax2 = fig.add_subplot(222)
    color_red = 'tab:red'
    color_blue = 'tab:blue'
    Re = IQarray[0,0,0,:].real
    Im = IQarray[0,0,0,:].imag
    line2, = ax2.plot(Re, color_red)
    ax3 = ax2.twinx()
    line3, = ax3.plot(Im, color_blue)

    '''
        imshow range-chirp , 
        imshow range-doppler 
    # '''
    # axIm = fig.add_subplot(223)
    # frame = Comfft[0,0,:,120:]
    # axIm = plt.imshow(abs(frame), aspect = 'auto', interpolation = 'catrom')
    # axIm = fig.add_subplot(223)

    '''
        imshow range-doppler(velocity)
    # '''
    # axDop = fig.add_subplot(224)
    # # veFrame = Vefft[0,0,:,:]
    # veFrame = Vefft[0,0,:20,22:42]
    # axDop = plt.imshow(veFrame, aspect= 'auto', origin='lower' , cmap='jet'
    # , extent=[-veFrame.shape[1]/2., veFrame.shape[1]/2., 0, veFrame.shape[0]-1], interpolation = 'catrom')
    # plt.colorbar(axDop)

    # ani = FuncAnimation(fig, animate, frames=470  ,interval = 200)

    plt.show()

def fftfunction():
    reIQarray = IQarray
    n = reIQarray.shape[3]
    # Comfft = np.fft.fftshift(np.fft.fft(reIQarray) / n)
    Comfft = np.fft.fft(reIQarray) / n
    print(Comfft.shape)
    Comfft = abs(Comfft)
    # Comfft = 20*np.log10(Comfft/32767)
    # Comfft = Comfft[:,:,:,range(n//2)]
    # plt.plot(Comfft[0,0,0,:])
    # plt.show()
    
    return Comfft

def fftVelocity():
    
    veFrame = Comfft[:,:,:,:]
    veFrame = np.swapaxes(veFrame,2,3)
    n = veFrame.shape[3]
    veFrame = np.fft.fftshift(np.fft.fft(veFrame)/n)
    veFrame = abs(veFrame)
    # veFrame = 20*np.log10(veFrame/32767)
    # print(veFrame.shape[3]/2)

    # cut2 = veFrame[:, :, :, :int(veFrame.shape[3]/2)]
    # cut1 = veFrame[:, :, :, int(veFrame.shape[3]/2):]


    # frame2 = np.concatenate((cut1,cut2), axis=3)
    
    # plt.imshow(frame2[0,0,:,:], aspect = 'auto', origin='lower', cmap='jet')
    return veFrame
    # return frame2

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



def main():

    global IQarray, Comfft, Vefft, Vdop

    name_count = 1
    Vefft_list = []
    f_name = glob.glob('D:/NestData/*.bin')

    print(f_name[0])
    
    for fname in f_name[:1]:
        
        IQarray = callBinfile(fname)
        print(IQarray.shape)
        # Comfft = fftfunction()
        # Comfft = movingAvg_OneD()
        # print(Comfft.shape)
        # Vefft = fftVelocity()
        # Vefft_real = Vefft.real
        # Vefft_imag = Vefft.imag
        # Vefft = np.concatenate((Vefft_real,Vefft_imag))

        # Vefft = np.swapaxes(Vefft, 0, 1)
        # Vefft_list.extend(Vefft)
        # Vefft = Vefft[:,:,:20,22:42]

        # Comfft = abs(Comfft)
        # Comfft = 20*np.log10(Comfft/32768)
        runGraphInitial()
        
    # Vefft_list = np.array(Vefft_list)
    # print(Vefft_list.shape)
    # Vefft_list = Vefft_list[:,:,:20,22:42]
    # np.save("D:/NestData/11-7-2019-64-chirp-16bit/pos_process_new/radar_data_reduce_all_real_imag.npy",Vefft_list) 
    
    '''
    save file to npy
    '''
    # np.save("D:/NestData/7-19-2019-new/signal_10_test_moving", veFrame)
    # np.save("Comfft", Comfft)
    
    
if __name__ == '__main__':
    main()  
