import numpy as np
import os
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft

def main():
    for fn in os.listdir('../../phonemes-wav/'):
         print(fn)
         if (fn != '.DS_Store'):
             print("../../phonemes-wav/" + fn)
             wav1, wav2 = wavfile.read("../../phonemes-wav/" + fn)
             curr_wav_fft = fft(wav2)
             print(curr_wav_fft.shape)


if __name__ == '__main__': # Main function
    main()
