import argparse
import requests
import os
import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow import keras
import base64
import json


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

url = 'http://192.168.137.100:8080/'

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

