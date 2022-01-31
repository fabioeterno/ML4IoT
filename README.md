# ML4IoT
Code for the homeworks of Machine Learning for Iot course of Politecnico di Torino academic year 2021/2022

## Approximate computing optimization techniques 

### Post-training quantization
The weights and activations of the model have been converted from floating point 32-bit to integer 8-bit after the training of the model in float32 format.
A representative dataset of input data has been passed to the tflite converter in order to allow him to compute the scaling factor and the offset parameters for both the weights and activations.

![image](https://user-images.githubusercontent.com/75701457/151764294-8613c90b-94f4-4a5a-bb6c-251c6f290bfb.png)

### Node-base pruning
In node-based pruning the number of nodes/channels is multiplied by a common value alpha between (0,1). In this way the overall number of weights of the neural network is reduced.

![image](https://user-images.githubusercontent.com/75701457/151766477-452a6661-aac1-4428-b1d3-f63a68b50144.png)

## RESTful API (and MQTT)

### Model Registry
The model registry application (HW3_ex1_Group3) is based on a RESTful API. The registry_client.py in the notebook folder sends with a POST HTTP request a tflite model to the registry_service.py in the raspberry pi folder, where the model is stored into a folder called "models". The client can retrieve the list of tflite models saved in the "models" folder with a GET HTTP request to the registry_service.py. The registry_client.py can also, with a POST HTTP request, specify the model to use to retrieving the inferences from the temperature and humidity sensor. Furthermore, the client can specify in the body of the request the temperature and humidity threshold to monitor. In case the prediction errore is above one of the two threshold a messagw with MQTT is published with topic "homework3/ex1/alarm". The monitoring_client.py made a subscription to that topic in order to receive as soon as possible the message.

### Edge-cloud collaborative inference
The fast_client.py is an application running on the edge device (the raspberry pi) implementing a simple keyword spotting recognition over the minispeech commands dataset. To minimize the energy consumption the audio files are preprocessed with a lower sampling frequency. When the score margin is under the 20% the input data is sent to the slow_service.py (the notebook) which plays the role of the cloud server and it retrieves the label predicted. The accuracy on the cloud server is higher but also the cost in terms of ebergy consumption for sending the data from the client is larger. The success checker policy implements this trade-off in order to reach the best possible accuracy on the edge device with the lowest possible power consumption.

![image](https://user-images.githubusercontent.com/75701457/151768449-7d0ca9b1-9f15-4433-9017-643f530a807a.png)
