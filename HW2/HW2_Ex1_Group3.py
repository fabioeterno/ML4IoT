import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow.lite as tflite
import argparse
import zipfile
import os 
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zlib


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default= "a", required = False, help='Version of exercise')
args = parser.parse_args()
vers = args.version



dataset_path = tf.keras.utils.get_file(
   "jena",
   "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip")
 
unzipped_dir = "jena_dataset/"
 
zip_ref = zipfile.ZipFile(dataset_path) # create zipfile object
zip_ref.extractall(unzipped_dir) # extract file to dir
zip_ref.close()
 
with open("jena_dataset/jena_climate_2009_2016.csv") as f:
    df = pd.read_csv(f)


column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)


input_width = 6
output_width = 3
LABEL_OPTIONS = 2


class WindowGenerator:
    def __init__(self, input_width, output_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.output_width = output_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :-self.output_width, :]

        labels = features[:, -self.output_width:, :]

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_width, 2])
        

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width + output_width,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total',
                initializer='zeros', shape=(2,))
        self.count = self.add_weight('count',
                initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred has shape (N, outwidth,2) like (32, 6, 2)
        # y_true has shape (N, outwidth,2) like (32, 6, 2)
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=0)
        error = tf.reduce_mean(error, axis=0)
        # error has shape (1, 2)
        self.total.assign_add(error)
        self.count.assign_add(1.)
        
        return
    
    def reset_state(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        
        return
    
    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        
        return result


if (vers == "a"):
    output_width= 3
else:
    output_width = 9


metrics = [MultiOutputMAE()]
generator = WindowGenerator(input_width, output_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)


def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]







pruning_params = {'pruning_schedule':
tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.30,
    final_sparsity=0.91,
    begin_step=len(train_ds)*5,
    end_step=len(train_ds)*15)
}

dir_mlp  = 'model_mlp_{}'.format(args.version)


if(args.version == "a"):

    alpha = 0.22

    MLP_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6, 2)),
        keras.layers.Dense(int(128*alpha), activation = 'relu', name = 'first_dense'),
        keras.layers.Dense(output_width*2, name = 'output'),
        keras.layers.Reshape((output_width, 2), input_shape=(output_width*2,))])


else:
    alpha = 0.125

    MLP_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(6, 2)),
        keras.layers.Dense(int(128*alpha), activation = 'relu', name = 'first_dense'),
        keras.layers.Dense(int(128*alpha), activation = 'relu', name = 'second_dense'),
        keras.layers.Dense(output_width*2, name = 'output'),
        keras.layers.Reshape((output_width, 2), input_shape=(output_width*2,))])

model_mlp = MLP_model
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_mlp = prune_low_magnitude(model_mlp, **pruning_params)
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]


model_mlp.compile(
    optimizer=tf.optimizers.Adam(),
    loss=tf.keras.metrics.mean_squared_error,
    metrics = metrics
    )

print(model_mlp.output_shape)

model_mlp.fit(train_ds, epochs = 20, validation_data = val_ds, callbacks = callbacks)
loss, error = model_mlp.evaluate(test_ds)

model_mlp = tfmot.sparsity.keras.strip_pruning(model_mlp)

run_model = tf.function(lambda x: model_mlp(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
model_mlp.save(dir_mlp, signatures=concrete_func)



# Post training quantization weight + activations
model = tf.keras.models.load_model(dir_mlp, compile=False)
input_shape = model.inputs[0].shape.as_list()
func = tf.function(model).get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_functions([func])


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

name_tflite_model = 'Group3_th_{}.tflite'.format(args.version)
name_tflite_model_c = 'Group3_th_{}.tflite.zlib'.format(args.version)

with open(name_tflite_model, 'wb') as f:
    f.write(tflite_model)

with open(name_tflite_model_c, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)


print("TFLite size: ", os.path.getsize(name_tflite_model_c)/1024, " Kb")