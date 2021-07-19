"""A library defining constants for the graph dumpers."""

import numpy
import tensorflow as tf

DICT_FIELD_NUMBERS = {
    'Const': 1,
    'Placeholder': 2,
    'Identity': 3,
    'BiasAdd': 4,
    'MaxPool': 5,
    'MatMul': 6,
    'Reshape': 7,
    'Relu': 8,
    'Conv2D': 9,
    'Add': 10,
    'ConcatV2': 11,
    'Mul': 12,
    'Maximum': 13,
    'LeakyReLU': 14,
    
    # In "tf2cpp", the same layer performs the matrix multiplication
    # and the matrix multiplication by batches.
    'BatchMatMul': 6,
    
    # "BatchMatMulV2" did not exist in Tensorflow 1.9. It exists in
    # Tensorflow 1.15.
    'BatchMatMulV2': 6
}

DICT_DTYPE_TENSORFLOW_DTYPE_TF2CPP = {
    tf.int32: 0,
    tf.float32: 1
}

DICT_DTYPE_TF2CPP_DTYPE_NUMPY = {
    0: numpy.int32,
    1: numpy.float32
}


