import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
import tensorflow as tf

def getModel():
    node_input_0 = tf.placeholder(tf.float32,
                                  shape=(1, 3, 8, 8),
                                  name='node_input_0')
    weights0 = tf.get_variable('weights0',
                              dtype=tf.float32,
                              initializer=tf.random_normal([3, 3, 8, 16], mean=0., stddev=0.1, dtype=tf.float32))
    biases0 = tf.get_variable('biases0', dtype=tf.float32, initializer=tf.random_normal([16],  mean=0., stddev=0.1, dtype=tf.float32))
        
    node_linear = tf.nn.conv2d(node_input_0, weights0, strides=[1, 1, 1, 1], padding='SAME')
    node_affine = tf.nn.bias_add(node_linear, biases0)
    y = tf.nn.relu(node_affine, name='layer0')
    
    weights1 = tf.get_variable('weights1',
                              dtype=tf.float32,
                              initializer=tf.random_normal([3, 3, 16, 32], mean=0., stddev=0.1, dtype=tf.float32))
    biases1 = tf.get_variable('biases1', dtype=tf.float32, initializer=tf.random_normal([32],  mean=0., stddev=0.1, dtype=tf.float32))
        
    node_linear = tf.nn.conv2d(y, weights1, strides=[1, 1, 1, 1], padding='SAME')
    node_affine = tf.nn.bias_add(node_linear, biases1)
    y = tf.nn.relu(node_affine, name='node_output')

    return {
            'list_names_nodes_input': ['node_input_0'],
            'list_nodes_input': [node_input_0],
            'list_names_nodes_output': ['node_output'],
            'list_nodes_output': [y],
            'tuple_saver_path_to_parameters': None
    }

        
def getInput(): 
   return [(np.random.rand(1, 3, 8, 8) - 0.5).astype(np.float32)]


