import numpy as np
import os
import tf2onnx
import onnx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
tensor_fmt = 'channels_last'
import tensorflow as tf

s = (16, 16,3)
inputs = tf.keras.Input(shape=s, name='input0', dtype=tf.float32)

nbf = 8 
x = tf.keras.layers.Conv2D(nbf, (3,3) , activation='linear',data_format=tensor_fmt, use_bias=True,bias_initializer="glorot_uniform",padding='same')(inputs)
x = tf.keras.layers.MaxPool2D(2,data_format=tensor_fmt)(x)
x = tf.keras.layers.Conv2D(nbf, (3,3) , activation='linear',data_format=tensor_fmt, use_bias=True,bias_initializer="glorot_uniform",padding='same')(x)

x0 = tf.keras.layers.Conv2D(nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x)
x0 = tf.keras.layers.Conv2D(nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x0)
x0 = x0 + x
x0 = tf.keras.layers.MaxPool2D(2,data_format=tensor_fmt)(x0)
x0 = tf.keras.layers.Conv2D(2*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x0)

x1 = tf.keras.layers.Conv2D(2*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x0)
x1 = tf.keras.layers.Conv2D(2*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x1)
x1 = x1 + x0
x1 = tf.keras.layers.MaxPool2D(2,data_format=tensor_fmt)(x1)
x1 = tf.keras.layers.Conv2D(4*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x1)

x2 = tf.keras.layers.Reshape((1,4*nbf*16//8*16//8))(x1)
y = tf.keras.layers.Dense(2)(x2)
model = tf.keras.Model(inputs=[inputs],outputs=y,name="cat_classifier")


X = np.linspace(-1.,1,np.prod(s)).reshape((1,)+s)
Y = model(X)

model_onnx , _ = tf2onnx.convert.from_keras(model,[tf.TensorSpec(shape=(1,)+s,name="input0")],opset=13)
onnx.save(model_onnx, "./tf2.onnx")
# print("Input\n",X)
print("Output\n",Y)

print("Model in tf2.onnx")

