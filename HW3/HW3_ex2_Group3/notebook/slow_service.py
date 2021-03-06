import numpy as np
import cherrypy
import json
import os
import base64
import time
import tensorflow as tf
import tensorflow.lite as tflite
from datetime import datetime

model_name = 'kws_dscnn_True'
LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

class Registry(object):
    exposed = True

    def GET(self, *path, **query):
        pass
    

    def POST(self, *path, **query):
        try:
            
            #read the body
            body = cherrypy.request.body.read()
            
            # # convert string into dictionary
            body = json.loads(body)  
            e = body.get('e')
            audio_str = e['v']
        
            # DECODING THE TENSOR FROM STRING INTO BASE64 BYTES
            audio_b64bytes = audio_str.encode()
            audio = base64.b64decode(audio_b64bytes)
            
            audio = tf.io.parse_tensor(audio, out_type=tf.float32)
           
            folder = os.path.join('{}.tflite'.format(model_name))
            interpreter = tflite.Interpreter(model_path=folder)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
                
            interpreter.set_tensor(input_details[0]['index'], audio)
            interpreter.invoke()

            predicted = interpreter.get_tensor(output_details[0]['index'])
            tmp = tf.math.softmax(predicted)*100
            print([ np.round(x, 2) for x in tmp.numpy()[0] ])
            index = np.argmax(predicted[0])
            print("predicted: ", LABELS[index], index)
            
            label = {"l" : str(index)}
            
            return json.dumps(label)

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
    cherrypy.config.update({'server.socket_host': '192.168.137.1'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
