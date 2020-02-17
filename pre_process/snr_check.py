import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

data_5x5 = np.load('D:/NestData/SNR_data2/radar_all/5_all.npy')
data_10x10 = np.load('D:/NestData/SNR_data2/radar_all/10_all.npy')
data_15x15 = np.load('D:/NestData/SNR_data2/radar_all/15_all.npy')
data_20x20 = np.load('D:/NestData/SNR_data2/radar_all/20_all.npy')

# print(data_5x5.shape)

# data_5x5 = abs(data_5x5)
# data_5x5 = 20*np.log10(data_5x5/32767)
# data_5x5 = data_5x5[:,:70]
# rms_noise_5x5 = np.sqrt(np.mean(data_5x5**2, axis=1 ))

# print(rms_noise_5x5)
# data_10x10 = abs(data_10x10)
# data_10x10 = 20*np.log10(data_10x10/32767)
# data_10x10 = data_10x10[:,:70]
# rms_noise_15x15 = np.sqrt(np.mean(data_10x10**2, axis=1 ))


# data_15x15 = abs(data_15x15)
# data_15x15 = 20*np.log10(data_15x15/32767)
# data_15x15 = data_15x15[:,:70]
# rms_noise_15x15 = np.sqrt(np.mean(data_15x15**2, axis=1 ))

data_20x20 = abs(data_20x20)
# data_20x20 = 20*np.log10(data_20x20/32767)
data_20x20 = data_20x20[:,:70]
rms_noise_20x20 = np.sqrt(np.mean(data_20x20**2, axis=1 ))

# p_10, _ = find_peaks(data_5x5[:10], distance=100)
# p_15, _ = find_peaks(data_15x15[:10], distance=100)
# p_10, _ = find_peaks(data_10x10[:10], distance=100)
p_20_0, _ = find_peaks(data_20x20[0,:10], distance=100)
p_20_1, _ = find_peaks(data_20x20[1,:15], distance=100)
p_20_2, _ = find_peaks(data_20x20[2,:20], distance=100)
p_20_3, _ = find_peaks(data_20x20[3,:25], distance=100)
p_20_4, _ = find_peaks(data_20x20[4,:30], distance=100)
p_20_5, _ = find_peaks(data_20x20[5,:35], distance=100)

# print(p_20,p_20,p_20,p_20)
# print(data_20x20[p_20_0])
print(data_20x20[0,p_20_0], data_20x20[1,p_20_1], data_20x20[2,p_20_2], data_20x20[3,p_20_3], data_20x20[4,p_20_4], data_20x20[5,p_20_5])

snr_signal_0 = 20*np.log10(data_20x20[0,p_20_0]/rms_noise_20x20[0])
snr_signal_1 = 20*np.log10(data_20x20[1,p_20_1]/rms_noise_20x20[1])
snr_signal_2 = 20*np.log10(data_20x20[2,p_20_2]/rms_noise_20x20[2])
snr_signal_3 = 20*np.log10(data_20x20[3,p_20_3]/rms_noise_20x20[3])
snr_signal_4 = 20*np.log10(data_20x20[4,p_20_4]/rms_noise_20x20[4])
snr_signal_5 = 20*np.log10(data_20x20[5,p_20_5]/rms_noise_20x20[5])

print("snr", snr_signal_0, snr_signal_1, snr_signal_2, snr_signal_3, snr_signal_4, snr_signal_5)

# print(snr_signal)
# plt.plot(data_20x20[0,:70])
# plt.plot(data_20x20[0,:70])
# plt.plot(data_20x20[0,:70])
# plt.plot(data_20x20)
# plt.plot(data_20x20)
# plt.plot(data_20x20)
plt.plot(data_20x20[0])
plt.plot(data_20x20[1])
plt.plot(data_20x20[2])
plt.plot(data_20x20[3])
plt.plot(data_20x20[4])
plt.plot(data_20x20[5])


plt.plot(p_20_0, data_20x20[0,p_20_0],"x")
plt.plot(p_20_1, data_20x20[1,p_20_1],"x")
plt.plot(p_20_2, data_20x20[2,p_20_2],"x")
plt.plot(p_20_3, data_20x20[3,p_20_3],"x")
plt.plot(p_20_4, data_20x20[4,p_20_4],"x")
plt.plot(p_20_5, data_20x20[5,p_20_5],"x")

# plt.plot(p_15, data_15x15[p_15],"x")
# plt.plot(p_10, data_10x10[p_10],"x")
# plt.plot(p_10, data_5x5[p_10],"x")
# plt.plot(data_15x15[0,:70])
# plt.plot(data_10x10[0,:70])
# plt.plot(data_5x5[0,:70])
# plt.plot(data_20x20[4,:70])
# plt.plot(data_20x20[5,:70])
# plt.plot(data_15x15[6,:70])

plt.show()

