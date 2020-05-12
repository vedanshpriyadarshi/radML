from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import csv
from pylab import *
import pylab
print('import done')

def make_data(n=10000,obj='without_obj'):
    fs = 2.6e6
    with open("desktop/SDR/{0}.csv".format(obj),"w",newline="") as k:
        for i in range(n):
            sdr = RtlSdr()
            samples = []
            sdr.sample_rate = 2.6e5
            sdr.center_freq = 431e6
            sdr.gain = 40
            sample = sdr.read_samples(512*1024)
            sdr.close()
            samples.append(sample.tolist())
            print(i)
            #ylim((-50, 40))
            plt.specgram(sample, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
            #plt.xlabel('Frequency (MHz)')
            #plt.ylabel('Relative power (dB)')
            #lt.yticks(np.arange(-40,28,4))
            #x1,x2,y1,y2 = plt.axis()
            #plt.axis((x1,x2,-50,50))

            
            plt.savefig('desktop/SDR/Data/431/metal/g{0}'.format(i),dpi=175, bbox_inches='tight')
            #plt.show()
            #plt.clf()
            

            
            




           
            #f, t, Sxx = signal.spectrogram(sample, fs, return_onesided=False)
            #plt.pcolormesh(t, f, Sxx)
            
            #plt.ylabel('Frequency [Hz]')
            #plt.xlabel('Time [sec]')
            #plt.axis('off')
            #plt.show()
        
            #plt.savefig("desktop/SDR/Spectrogram.png".format(i), bbox_inches='tight')
            #samples.append(sample.tolist())
            print('!')
            writer = csv.writer(k)
            writer.writerows(samples)
        k.close()

make_data(300,'without_obj_1')