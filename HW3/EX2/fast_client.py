import argparse
import os
import numpy as np
import os
import tensorflow as tf
import tensorflow.lite as tflite
import sys
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import numpy as np
import base64
import requests

model_name = 'kws_dscnn_True'
num_frames = 49
num_coefficients = 10

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)

num_samples = len(filenames)

#total = 8000
#train_files = filenames[:int(total*0.8)]
#val_files = filenames[int(total*0.8): int(total*0.9)]
#test_files = filenames[int(total*0.9):]

LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1
        
        self.num_frames = (sampling_rate - frame_length) // frame_step + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)        
        audio_binary = tf.io.read_file(file_path)                
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):

        zero_padding = tf.zeros(tf.abs([self.sampling_rate] - tf.shape(audio)), dtype=tf.float32)

        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio
    

    def get_spectrogram(self, audio):       
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]
        
        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

#MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
#        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
#        'num_coefficients': 10}

MFCC_OPTIONS = {'frame_length': 600, 'frame_step': 300, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}


options = MFCC_OPTIONS
strides = [2, 1]

generator = SignalGenerator(LABELS, 15000, **options)

#train_ds = generator.make_dataset(train_files, True)
#val_ds = generator.make_dataset(val_files, False)
dataset = generator.make_dataset(filenames, False)

units = 8

folder = '{}.tflite'.format(model_name)
interpreter = tflite.Interpreter(model_path=folder)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#print(output_details)    
dataset = dataset.unbatch().batch(1)

correct = 0.0
total = 0.0

for elem in dataset:
    
    #tf.reshape(mfccs, [1,  self.num_frames, self.num_coefficients, 1])
    #input_tensor = tf.reshape(elem[0], [1,  num_frames, num_coefficients, 1])
    interpreter.set_tensor(input_details[0]['index'], elem[0])
    interpreter.invoke()

    predicted = interpreter.get_tensor(output_details[0]['index'])
    #print(predicted[0])
    tmp = tf.math.softmax(predicted)*100
    #print(tmp.numpy()[0])
    print([ np.round(x, 2) for x in tmp.numpy()[0] ])
    index = np.argmax(predicted[0])
    print("predicted: ", LABELS[index], index)
    print("true value: ", LABELS[elem[1][0]], elem[1][0].numpy() )
    
    print(elem[0])
    
    # ENCODING THE MODEL FROM BASE64 BYTES INTO STRING
    elem_serial = tf.io.serialize_tensor(elem[0])
    audio_b64bytes = base64.b64encode(elem_serial.numpy())
    audio_string = audio_b64bytes.decode() 
    
    print(audio_string)
    
    url = 'http://127.0.0.1:8080/'
    body = {'audio': audio_string}
    
    # Conversion in json of the body
    r = requests.post(url, json=body)
    if r.status_code == 200:
        print(r)
    else:
        print('Error:', r.status_code)
    
    sys.exit()
    
    # the index predicted is equal to the index of the true label in test_ds elem[1][0]
    if index==elem[1][0]:
        correct+=1.0
    total +=1.0
    


print()    
print("TFLite quantized accuracy: " + str('{0:.10f}'.format(correct/total)))    
print()

sys.exit()





















































#model_name = args.model[0]
model_name = '2_cnn'


# Converting saved model to TFLite model
converter = tf.lite.TFLiteConverter.from_saved_model('models_source/{}'.format(model_name))
tflite_model = converter.convert()

# Saving the TFLite model on disk
name_tflite_model = os.path.join('./models_source/', '{}.tflite'.format(model_name))
with open(name_tflite_model, 'wb') as f:
    f.write(tflite_model)

# ENCODING THE MODEL FROM BASE64 BYTES INTO STRING
model_b64bytes = base64.b64encode(tflite_model)
model_string = model_b64bytes.decode() 
#print(model_string)

url = 'http://127.0.0.1:8080/'

# TESTING ADD PATH RESPONSE
url_add = os.path.join(url,'add/')
body = {'model': model_name, 'model_string': model_string}
# Conversion in json of the body
r = requests.post(url_add, json=body)
if r.status_code == 200:
    print(r)
else:
    print('Error:', r.status_code)

# TESTING LIST PATH RESPONSE    
url_list = os.path.join(url,'list/')
r = requests.get(url_list)

if r.status_code == 200:
    print(r)
    print("Registered models: ", r.content.decode())
else:
    print('Error:', r.status_code)


# TESTING PREDICT RESPONSE    
tthres = 0.2
hthres = 0.3
body = {'model': model_name, 'tthres':tthres, 'hthres':hthres}
url_predict = os.path.join(url, 'predict/')
r = requests.post(url_predict, json = body)

if r.status_code == 200:
    print(r)
    print("Pred: ", r.content.decode())
else:
    print('Error:', r.status_code)

