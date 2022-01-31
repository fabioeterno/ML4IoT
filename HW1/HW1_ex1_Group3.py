import tensorflow as tf
import os
import time
import datetime
import argparse
import pandas as pd
import numpy as np

#input example
# %Run HW1_ex1_Group3.py --input dht.csv --output dhttf.tfrecord --normalize False

#create argument parser
parser = argparse.ArgumentParser(description='definition of parameters')
parser.add_argument('--input', default = 'dht.csv',
                    help='input csv file with recordings',
                    required=True)
parser.add_argument('--output', default = 'dhttf.tfrecord',
                    help='output tfrecords file',
            required=True)
parser.add_argument('--normalize', default = False,
                    help='flag for normalize input')
args = parser.parse_args()



# Taking the parameters from command line
input_file = args.input
output_file = args.output
flg_normalization = args.normalize

# creating a table using pandas frame
d_types={'date':str, 'time':str, 'temperature':int, 'humidity':int}
columns = ['date', 'time', 'temperature', 'humidity']
df = pd.read_csv(input_file, sep=',', dtype=d_types, header=None, names=columns)


#"%d/%m/%Y %H:%M:%S"
#convert datetime format to posix 

dt = df['date']+ " " +df['time']
for e in range(len(dt)):
    dt.loc[e] = time.mktime(datetime.datetime.strptime(dt.loc[e], "%d/%m/%Y %H:%M:%S").timetuple())

#example output for timestamp
#1635861286.0

#create numpy array for tfrecord
filename = output_file #'/home/pi/WORK_DIR/homework1/dhttf.tfrecord'
x1 = dt.to_numpy()
x2 = df['temperature'].to_numpy()
x3 = df['humidity'].to_numpy()

#boundaries taken from datasheet
min_temp = 0
max_temp = 50
min_hum = 20
max_hum = 90

#flag for normalization
if (flg_normalization):
    for i in range (len(x2)):
        x2[i] = (x2[i] - min_temp) / (max_temp - min_temp)   #for temperature
        x3[i] = (x3[i] - min_hum) / (max_hum - min_hum)    #for humidity

with tf.io.TFRecordWriter(filename) as writer:
    for i in range(len(df)):
        #create features for tfrecord
        x1_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[x1[i]]))
        x2_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[x2[i]]))
        x3_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[x3[i]]))
        mapping = {'float': x1_feature,
                    'float' : x2_feature,
                    'float' : x3_feature}            
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
        writer.write(example.SerializeToString())
    
    
#print("CSV file size: ", str(os.path.getsize(input_file))+str("B"))
#print("TfRecord file size: ", str(os.path.getsize(output_file)) + str("B"))
        
print(str(os.path.getsize(output_file)) + str("B"))

