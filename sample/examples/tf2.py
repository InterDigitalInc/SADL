import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
import tensorflow as tf

def getModel():
   input_0 = tf.keras.Input(shape=(3, 8, 8), name='input0')
   y = tf.keras.layers.Conv2D(16,3, strides=(1, 1), activation='relu',use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='glorot_uniform',
                              padding='same',
                              data_format='channels_last')(input_0)
   y = tf.keras.layers.Conv2D(32,3, strides=(1, 1), activation='relu',use_bias=True,
                              kernel_initializer='glorot_uniform',
                              bias_initializer='glorot_uniform',
                              padding='same',
                              data_format='channels_last')(y)                           
   model = tf.keras.Model(inputs=[input_0],
                          outputs=y,
                          name='test_tf2')
   return model
        
def getInput(): 
   return [(np.random.rand(1, 3, 8, 8) - 0.5).astype(np.float32)]

