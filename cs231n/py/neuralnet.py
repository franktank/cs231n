import numpy as np
import os
import scipy.io.wavfile as wavfile
from scipy.fftpack import fft

class NeuralNet(object):
    def __init__(self, dimensionality, hlsize, num_classes):
        self.parameters = {}
        
