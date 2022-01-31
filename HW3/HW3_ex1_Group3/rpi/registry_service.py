import numpy as np
import cherrypy
import json
import os
import base64
import math
#import RPi.GPIO as GPIO
import time
#from time import time
import adafruit_dht
import board
import tensorflow as tf
import tensorflow.lite as tflite
from datetime import datetime
from DoSomething import DoSomething

destination = './models'
mean = np.array([9.107597, 75.904076])
std = np.array([8.654227, 16.557089])

class Registry(object):
    exposed = True

    def GET(self, *path, **query):
        try:
            if path[0] == 'list':            
                models_registered = [_ for _ in os.listdir(destination) if _.endswith(".tflite")]
                models_registered_json = json.dumps(models_registered)
                return models_registered_json
        except Exception as E:
            raise cherrypy.HTTPError(400, 'Error')

    def POST(self, *path, **query):
        try:
            
            if len(path) > 1:
                raise cherrypy.HTTPError(400, 'Wrong path')
            
            #read the body
            body = cherrypy.request.body.read()
            
            # # convert string into dictionary
            body = json.loads(body)
            model_name = body.get('model')
            
            # ADD MODEL TO MODELS FOLDER
            if path[0] == 'add':
                if not os.path.exists('./models'):
                    os.makedirs('./models')        
                model_str = body.get('model_string')
                # DECODING THE MODEL FROM STRING INTO BASE64 BYTES
                model_b64bytes = model_str.encode()
                model = base64.b64decode(model_b64bytes)
                
                # SAVING THE TFLite MODEL ON DISK
                folder = os.path.join('./models/', '{}.tflite'.format(model_name))
                with open(folder, 'wb') as f:
                    f.write(model)          
            
            if path[0] == 'predict':
                
                # CREATING A PUBLISHER FOR HUMIDITY/TEMPERATURE ALARM
                publisher = DoSomething("publisher")
                publisher.run()                

                model_str = body.get('model')
                tthres = body.get('tthres')
                hthres = body.get('hthres')

                  
                dht_device = adafruit_dht.DHT11(board.D27)
                period = 100
                freq = 1
                
                folder = os.path.join('./models/', '{}.tflite'.format(model_str))
                interpreter = tflite.Interpreter(model_path=folder)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                data = np.zeros([6,2], dtype = np.float32)
                
                i = 0
                while(i!=6):
                    try:
                        
                        time.sleep(freq)
                        temperature =dht_device.temperature
                        humidity = dht_device.humidity
                        data[i][0] = temperature
                        data[i][1] = humidity
                        print(data[i])
                        if(math.isnan(float(data[i][0]))):
                            i = i
                        else:
                            i= i+1
                    except Exception as e:
                        print("misread")
                        print(str(e))


                time.sleep(freq)


                temperature =dht_device.temperature
                humidity = dht_device.humidity
                label = np.array([temperature, humidity], dtype = np.float32)

                #normalize
                data = (data - mean) / (std + 1.e-6)

                input_data = tf.constant(data, shape= [1,6,2], dtype = tf.float32)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                predicted = interpreter.get_tensor(output_details[0]['index'])

                t_pred = abs(predicted[0][0] - label[0])
                h_pred = abs(predicted[0][1] - label[1])
                e = []
                bt = int(round(datetime.now().timestamp()))

                # TEMPERATURE CHECK
                if (t_pred > tthres):
                    
                    now = datetime.now()
                    # timestampt = now.strftime("(%d/%m/%Y %H:%M:%S)")
                    now = int(round(datetime.now().timestamp()))
                    tt = {"n": "temperature",
                        "u":"Cel", 
                        "t":now - bt, 
                        "v":str(t_pred)}
                    e.append(tt)                    
                    
                    # SENDING TEMPERATURE ALARM
                    temp_alarm = datetime.now().strftime('(%d-%m-%y %H:%M:%S)') \
                                 + ' Temperature Alert: Predicted={:.1f}°C Actual={:.1f}°C'.format(predicted[0][0], label[0])
                    temp_alarm_json = json.dumps({'alarm': temp_alarm})                    
                    publisher.myMqttClient.myPublish ("/homework3/ex1/alarm", temp_alarm_json)                    
                
                # HUMIDITY CHECK 
                if (h_pred > hthres):
                    now = datetime.now()
                    # timestamph = now.strftime("(%d/%m/%Y %H:%M:%S)")
                    now = int(round(datetime.now().timestamp()))
                    ht = {"n": "humidity",
                        "u":"%", 
                        "t":now - bt, 
                        "v":str(h_pred)}
                    e.append(ht)
                    
                    # SENDING HUMIDITY ALARM
                    hum_alarm = datetime.now().strftime('(%d-%m-%y %H:%M:%S)') \
                                 + ' Humidity Alert: Predicted={:.1f}% Actual={:.1f}%'.format(predicted[0][1], label[1])
                    hum_alarm_json = json.dumps({'alarm': hum_alarm})                    
                    publisher.myMqttClient.myPublish ("/homework3/ex1/alarm", hum_alarm_json)                    
                
                # STOPPING PUBLISHER OF ALARM
                publisher.end()
                
                if e:
                    now = datetime.now()
                    timestamp = now.strftime("(%d/%m/%Y %H:%M:%S)")
                    body = {"bn" : "raspberrypi.local" ,
                        "bt" : timestamp,
                        "e" : e
                    }
                
                    # convertion to json
                    return json.dumps(body)

        except Exception as E:
            print(E)
            raise cherrypy.HTTPError(400, E)
            
    def PUT(self, *path, **query):  
        pass
    
    def DELETE(self, *path, **query):
        pass 

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Registry(), '', conf)
    cherrypy.config.update({'server.socket_host': '192.168.137.100'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()