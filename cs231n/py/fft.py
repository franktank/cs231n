import numpy as np
import os
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft

def main():
    X_data = np.zeros(shape=(43,46))
    Y_label = np.zeros(shape=(43))
    i = 0
    for fn in os.listdir('../../phonemes-wav/'):
         if (fn != '.DS_Store'):
             print("../../phonemes-wav/" + fn)
             wav1, wav2 = wavfile.read("../../phonemes-wav/" + fn)
             Y_label[i] = i
             curr_wav_fft = fft(wav2)
             X_data[i,0] = curr_wav_fft[75].real[0]
             X_data[i,1] = curr_wav_fft[75].real[1]
             X_data[i,2] = curr_wav_fft[100].real[0]
             X_data[i,3] = curr_wav_fft[100].real[1]
             X_data[i,4] = curr_wav_fft[200].real[0]
             X_data[i,5] = curr_wav_fft[200].real[1]
             X_data[i,6] = curr_wav_fft[300].real[0]
             X_data[i,7] = curr_wav_fft[300].real[1]
             X_data[i,8] = curr_wav_fft[400].real[0]
             X_data[i,9] = curr_wav_fft[400].real[1]
             X_data[i,10] = curr_wav_fft[600].real[0]
             X_data[i,11] = curr_wav_fft[600].real[1]
             X_data[i,12] = curr_wav_fft[800].real[0]
             X_data[i,13] = curr_wav_fft[800].real[1]
             X_data[i,14] = curr_wav_fft[1000].real[0]
             X_data[i,15] = curr_wav_fft[1000].real[1]
             X_data[i,16] = curr_wav_fft[1200].real[0]
             X_data[i,17] = curr_wav_fft[1200].real[1]
             X_data[i,18] = curr_wav_fft[1400].real[0]
             X_data[i,19] = curr_wav_fft[1400].real[1]
             X_data[i,20] = curr_wav_fft[1600].real[0]
             X_data[i,21] = curr_wav_fft[1600].real[1]
             X_data[i,22] = curr_wav_fft[1800].real[0]
             X_data[i,23] = curr_wav_fft[1800].real[1]
             X_data[i,24] = curr_wav_fft[2000].real[0]
             X_data[i,25] = curr_wav_fft[2000].real[1]
             X_data[i,26] = curr_wav_fft[2200].real[0]
             X_data[i,27] = curr_wav_fft[2200].real[1]
             X_data[i,28] = curr_wav_fft[2400].real[0]
             X_data[i,29] = curr_wav_fft[2400].real[1]
             X_data[i,30] = curr_wav_fft[2600].real[0]
             X_data[i,31] = curr_wav_fft[2600].real[1]
             X_data[i,32] = curr_wav_fft[2800].real[0]
             X_data[i,33] = curr_wav_fft[2800].real[1]
             X_data[i,34] = curr_wav_fft[3000].real[0]
             X_data[i,35] = curr_wav_fft[3000].real[1]
             X_data[i,36] = curr_wav_fft[3200].real[0]
             X_data[i,37] = curr_wav_fft[3200].real[1]
             X_data[i,38] = curr_wav_fft[3500].real[0]
             X_data[i,39] = curr_wav_fft[3500].real[1]
             X_data[i,40] = curr_wav_fft[4000].real[0]
             X_data[i,41] = curr_wav_fft[4000].real[1]
             X_data[i,42] = curr_wav_fft[4500].real[0]
             X_data[i,43] = curr_wav_fft[4500].real[1]
             X_data[i,44] = curr_wav_fft[5000].real[0]
             X_data[i,45] = curr_wav_fft[5000].real[1]
             i += 1
    print(X_data)


if __name__ == '__main__': # Main function
    main()
