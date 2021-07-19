"""
/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2021, ITU/ISO/IEC
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # set tensorflow debug info level
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # set tensorflow debug info level
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'


import argparse
import importlib
import json
import numpy as np
import sys
import tensorflow as tf

# this is important to have determinitic random 
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)

# get py from current dir
sys.path.append(os.getcwd())

# for utest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../tests_data')


# limit accuacry of the output
def quantizer_inputs_n_bits(inputs, quantizer):  
    scale = np.math.pow(2, quantizer)
    list_inputs_quantized = []
    for i in range(len(inputs)):        
        # Only input Numpy array of data-type float can
        # be quantized.
        if not np.issubdtype(inputs[i].dtype, np.floating):
            raise TypeError('`inputs[{}].dtype` is not smaller than `np.float` in type-hierarchy.'.format(i))
        list_inputs_quantized.append(np.round(inputs[i]*scale)/scale)
    return list_inputs_quantized
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test py2cpp conversion utest',
                                     usage='NB: force run on CPU')
    parser.add_argument('--input_py',
                        action='store',
                        nargs='?',
                        type=str,
                        help='name of the py file for testing')
    parser.add_argument('--output',
                        action='store',
                        nargs='?',
                        type=str,
                        help='name of model binary file')
    parser.add_argument('--output_results',
                        action='store',
                        nargs='?',
                        type=str,
                        help='name of results file')
                        
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    np.random.seed(42)
    
    if args.input_py is not None:
        test = importlib.import_module(args.input_py, package=None)
    else:
        raise('[ERROR] You should specify a py file') 
        quit()
        
        
    # The inference depends on the version of Tensorflow.
    if tf.__version__.startswith('2'): 
        import model_dumper_tf2 as md
        tf.random.set_seed(42)
        print("[INFO] TF 2 converter")
        model = test.getModel()
        inputs = test.getInput()   
        inputs = quantizer_inputs_n_bits(inputs, 10)     
        if args.verbose:  print(model.summary())     
        md.dumpModel(model,inputs,
                     args.output,
                     args.verbose)
        if args.output_results is not None:
          output = model(inputs)
                     
    elif tf.__version__.startswith('1'):
        import model_dumper_tf1 as mdtf1
        tf.set_random_seed(42)
        print("[INFO] TF 1 converter")

        # `dict_model` provides the input tensors, the input tensor names,
        # the output tensors, and the output tensor names.
        inputs = test.getInput()
        inputs = quantizer_inputs_n_bits(inputs, 10)     
        dict_model = test.getModel()
         
        feed_dict = {}
        if len(dict_model['list_nodes_input']) != len(inputs):
            raise ValueError('`len(dict_model[\'list_nodes_input\']) is not equal to `len(inputs)`.`')
        
        # `feed_dict` associates each key input node
        # to its value Numpy array.
        for i in range(len(inputs)):
            feed_dict[dict_model['list_nodes_input'][i]] = inputs[i]
        with tf.Session() as sess:
            tuple_saver_path_to_parameters = dict_model['tuple_saver_path_to_parameters']
            if tuple_saver_path_to_parameters is None:
                tf.global_variables_initializer().run()
            else:
                tuple_saver_path_to_parameters[0].restore(sess,
                                                          tuple_saver_path_to_parameters[1])
            list_outputs = sess.run(
                dict_model['list_nodes_output'],
                feed_dict=feed_dict
            )
            mdtf1.convert_vars_to_constants_write_graph(sess,
                                                        dict_model['list_names_nodes_input'],
                                                        dict_model['list_names_nodes_output'],
                                                        dict_model['list_nodes_input'],
                                                        args.output,
                                                        args.verbose)
        if args.output_results is not None:
          output = list_outputs[0]
    else:
       raise('[ERROR] unknown framework type') 
                                                 

   
    # // results file
    # // nb inputs [int]
    # // for all input: 
    # //   input dim size [int] # between 1 and 4
    # //   for all dim in input dim size
    # //      dim[k] [int] 
    # //   input [float[nb elts]]
    # // nb output [int] # currently == 1
    # // for all output: 
    # //   output dim size [int] # between 1 and 4
    # //   for all dim in output dim size
    # //      dim[k] [int] 
    # //   output [float[nb_elts]]
    if args.output_results is not None:
      with open(args.output_results, 'w') as f:
        f.write(str(len(inputs)) + '\n')
        for input in inputs:
          if input.shape[0] != 1: raise('[ERROR] inputs should include a batch size of size=1.')
          if len(input.shape)<=1 or input.shape[0] != 1:                 
            print('[WARN] inputs should include batch size of size=1.')
            f.write('2' + '\n' +'1 ') # force batch 1
          else:
            f.write(str(len(input.shape)) + '\n') 
          for i in input.shape: f.write('{} '.format(i))
          f.write('\n')
          for x in np.nditer(input): f.write('{} '.format(x))
          f.write('\n')    
        f.write('1\n') 
        f.write(str(len(output.shape)) + '\n')
        for i in output.shape: f.write('{} '.format(i))
        f.write('\n')
        for x in np.nditer(output): f.write('{} '.format(x))
        f.write('\n')
      print('[INFO] results file in ', args.output_results)

    


