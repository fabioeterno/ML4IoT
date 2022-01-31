import tensorflow as tf
import os
import time
import zipfile
from subprocess import Popen
import math
import numpy as np


Popen('sudo sh -c "echo performance >'
     '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
     shell=True).wait()

#function for computing stft + mfcc

def compute_mfcc(audio, frame_length, frame_step, f = 16000,  num_mel_bins=40, lower_frequency = 20, upper_frequency = 4000, sampling_rate = 16000, matrix=[]):
    
    start_time = time.time()
    tf_audio, rate = tf.audio.decode_wav(audio) 
    tf_audio = tf.squeeze(tf_audio, 1)

    stft = tf.signal.stft(tf_audio,
                          frame_length=frame_length,
                          frame_step=frame_step,
                          fft_length=frame_length)

    spectrogram = tf.abs(stft)
    num_spectrogram_bins = spectrogram.shape[-1] # y axis of mel spectrogram

    #precomputed linear to mel weight matrix
    linear_to_mel_weight_matrix = matrix
    mel_spectrogram = tf.tensordot(
        spectrogram,
        linear_to_mel_weight_matrix,
        1)
        
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))
        
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
         
    # mfccs allows to select only the first portion of the mel spectrogram
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10]

    time_spent = time.time()-start_time
    return mfccs, time_spent


def SNR(mfcc_s, mfcc_f):
    SNR = 20*math.log10(np.linalg.norm(mfcc_s)/np.linalg.norm(mfcc_s-mfcc_f+10**(-6)))
    return SNR


def compute_matrix(num_mel_bins=40, num_spectrogram_bins=129, sampling_rate=16000, lower_frequency=20, upper_frequency=4000):
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
         num_mel_bins,
         num_spectrogram_bins,
         sampling_rate, # 16000
         lower_frequency,
         upper_frequency
        )
    return linear_to_mel_weight_matrix


def func(inputpath,s = 128, l = 256, f = 16000, num_mel_bins=40, lower_frequency = 20, upper_frequency = 4000, sampling_rate = 16000, num_spectrogram_bins = 129):

    n = 0
    matrix = compute_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate, lower_frequency, upper_frequency)
    
    total_time = 0

    for i in os.listdir(inputpath):
        
        audio = tf.io.read_file(os.path.join(inputpath, i))
        start_time = time.time()
        #compute mfcc and time spent
        mfcc, time_spent = compute_mfcc(audio, l, s, f, num_mel_bins, lower_frequency, upper_frequency, sampling_rate, matrix = matrix)
        
        total_time = total_time + time_spent
        n = n + 1
        
    avg = (total_time/n)*1000

    return mfcc, avg


# Routine for extracting file from a zipped folder
filename = "yes_no.zip"

zip_ref = zipfile.ZipFile(filename) # create zipfile object
zip_ref.extractall('yes_no_Data_/') # extract file to dir
zip_ref.close() # close file


inputpath = "yes_no_Data_/"
mfcc_slow, avg_slow = func(inputpath)
print("MFCC slow = ", str(round(avg_slow))+ " ms")

# New pre-processing pipeline
mfcc_fast, avg_fast = func(inputpath, s = 128, l =256, f = 16000, num_mel_bins = 32, sampling_rate = 10000,
                            lower_frequency = 20, upper_frequency = 4000)

print("MFCC fast = ", str(round(avg_fast))+ " ms")
print("SNR = ", str(round(SNR(mfcc_slow, mfcc_fast), 2)) +" dB")
