import cherrypy
import json
import os
import base64

destination = './models'

class Registry(object):
    exposed = True

    def GET(self, *path, **query):
        try:
            if path[0] == 'list':            
                models_registered = os.listdir(destination)
                models_registered_json = json.dumps(models_registered)
                print(models_registered_json)
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
            model_name = body.get('name')
            
            # ADD MODEL TO MODELS FOLDER
            if path[0] == 'add':
                if not os.path.exists('./models'):
                    os.makedirs('./models')        
                model_str = body.get('model')
                
                # DECODING THE MODEL FROM STRING INTO BASE64 BYTES
                model_b64bytes = model_str.encode()
                model = base64.b64decode(model_b64bytes)
                
                # SAVING THE TFLite MODEL ON DISK
                folder = os.path.join('./models/', '{}.tflite'.format(model_name))
                with open(folder, 'wb') as f:
                    f.write(model)          
            
            if path[0] == 'predict':            
                model_str = body.get('model')
                tthres = body.get('tthres')
                hthres = body.get('hthres')

                response_body = []
                # 
                # TODO: run prediction here, try to pass json not list 
                ##
                #  if tthres_pred > tthres:
                #      response_body = response_body.append("{} Temperature Alert: Predicted={}°C Actual={}°C".format(<Timestamp>, <pred>, <actual>))
                #  if hthres_pred > hthres:
                #      response_body = response_body.append("{} Humidity Alert: Predicted={}% Actual={}%".format(<Timestamp>, <pred>, <actual>))
        except Exception as E:
            raise cherrypy.HTTPError(400, 'Error')
            
    def PUT(self, *path, **query):  
        pass
    
    def DELETE(self, *path, **query):
        pass 

if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Registry(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()