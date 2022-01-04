import argparse
import os
import numpy as np
import os
import tensorflow as tf
import tensorflow.lite as tflite
import sys
import pandas as pd
import zlib
import tensorflow_model_optimization as tfmot


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model name')
args = parser.parse_args()


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)



zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)

num_samples = len(filenames)

total = 8000

# validation can be used only for hyperparameter tuning or earlystopping training    

train_files = open("kws_train_split.txt", "r")
train_files = tf.convert_to_tensor(train_files.read().splitlines())
val_files = open("kws_val_split.txt", "r")
val_files = tf.convert_to_tensor(val_files.read().splitlines())
test_files = open("kws_test_split.txt", "r")
test_files = tf.convert_to_tensor(test_files.read().splitlines())

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
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
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


STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
#MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
#        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
#        'num_coefficients': 10}

MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}


#if args.mfcc is True:
options = MFCC_OPTIONS
strides = [2, 1]
#else:
#    options = STFT_OPTIONS
#    strides = [2, 2]


generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)

units = 8

if args.version == 'a':
    MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
    # dscnn
    model_name = "Group3_kws_a.tflite.zip"  
    model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(units=units)  
    ])
if args.version == 'b':
    MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
    # dscnn + magnitude base pruning (unstructured pruning) 
    model_name = "Group3_kws_b.tflite.zip"  
    alpha = 0.99
    model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[3, 3], strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                tf.keras.layers.BatchNormalization(momentum=0.1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(units=units)  
    ]) 

    #pruning_params = {'pruning_schedule':
    #                         tfmot.sparsity.keras.PolynomialDecay(
    #                            initial_sparsity=0.30,
    #                            final_sparsity=0.8,
    #                            begin_step=len(train_ds)*5,
    #                            end_step=len(train_ds)*15
    #                        )
    #                  tfmot.sparsity.keras.ConstantSparsity(
    #                               target_sparsity=0.8, begin_step=0, end_step=-1, frequency=100
    #                            )
    #                 }
    #prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    #model = prune_low_magnitude(model, **pruning_params)
    

# Compile the model
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
metrics = [tf.metrics.SparseCategoricalAccuracy()]



# Train the model
if args.version == 'a':
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_ds, epochs=30, validation_data=val_ds)   
    print(model.summary())
    # Evaluate the model
    loss, error = model.evaluate(test_ds)
if args.version == 'b':
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(train_ds, epochs=20, validation_data=val_ds)
    print(model.summary())
    # Evaluate the model
    loss, error = model.evaluate(test_ds)  
if args.version == 'c':
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    input_shape = [32, 49, 10, 1]
    model.build(input_shape)
    model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
    print(model.summary())
    # Evaluate the model
    loss, error = model.evaluate(test_ds)
    model = tfmot.sparsity.keras.strip_pruning(model)
    
# Weights clustering
if args.version == 'a':
    clustered_model = tfmot.clustering.keras.cluster_weights(
                model,
                number_of_clusters=25,
                cluster_centroids_init = tfmot.clustering.keras.CentroidInitialization.LINEAR
    )
#if args.version == 'b':
#        clustered_model = tfmot.clustering.keras.cluster_weights(
#                model,
#                number_of_clusters=25,
#                cluster_centroids_init = tfmot.clustering.keras.CentroidInitialization.LINEAR
#    )
    

model = tfmot.clustering.keras.strip_clustering(clustered_model)

saved_model_dir = os.path.join('.', 'models', '{}'.format(model_name))
model.save(saved_model_dir)

# Converting saved model to TFLite model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Saving the TFLite model on disk
name_tflite_model = '{}_original'.format(model_name)
with open(name_tflite_model, 'wb') as f:
    f.write(tflite_model)

print()
print('Model accuracy:', error)
print("tflite model size: ", os.path.getsize(name_tflite_model))
print()

def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]

# Post training quantization weight + activations
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

# Compressing the TFLite quantized model    
name_tflite_model_quant = model_name
with open(name_tflite_model_quant, 'wb') as f:
    tflite_compressed = zlib.compress(tflite_quant_model)
    f.write(tflite_compressed)
    
# Loading TFLite model    
tflite_decompressed = open(name_tflite_model_quant, 'rb').read()
tflite_decompressed = zlib.decompress(tflite_decompressed)    
    
#interpreter = tflite.Interpreter(model_path=name_tflite_model_quant)
interpreter = tflite.Interpreter(model_content=tflite_decompressed)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_ds = test_ds.unbatch().batch(1)

correct = 0.0
total = 0.0

for elem in test_ds:

    interpreter.set_tensor(input_details[0]['index'], elem[0])
    interpreter.invoke()

    predicted = interpreter.get_tensor(output_details[0]['index'])
    index = np.argmax(predicted[0])
    
    # the index predicted is equal to the index of the true label in test_ds elem[1][0]
    if index==elem[1][0]:
        correct+=1.0
    total +=1.0
print()    
print("TFLite quantized accuracy: " + str('{0:.10f}'.format(correct/total)))    
print("tflite quantized model size: ", os.path.getsize(name_tflite_model_quant))
print()