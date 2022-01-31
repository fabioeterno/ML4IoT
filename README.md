# ML4IoT
Code for the homeworks of Machine Learning for Iot course of Politecnico di Torino academic year 2021/2022

## Approximate computing optimization techniques: 

### Post-training quantization
The weights and activations of the model have been converted from floating point 32-bit to integer 8-bit after the training of the model in float32 format.
A representative dataset of input data has been passed to the tflite converter in order to allow him to compute the scaling factor and the offset parameters for both the weights and activations.

![image](https://user-images.githubusercontent.com/75701457/151764294-8613c90b-94f4-4a5a-bb6c-251c6f290bfb.png)

### Node-base pruning
