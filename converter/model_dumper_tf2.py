"""A library defining functions dumping models in Tensorflow 2.x.
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

import json
import numpy as np
import struct

import constants.constants as csts
 
# file format:    
# MAGIC: SADL0001 [char[8]]
# type_model [int32_t] 0:int32, 1:float, 2:int16
# nb_layers [int32_t]
# nb_inputs [int32_t]
# inputs_id [int32_t[nb_inputs]]
# nb_outputs [int32_t]
# outputs_id [int32_t[nb_outputs]]
# (for all layers:)
#  layer_id [int32_t]
#  op_id    [int32_t]
#  name_size [int32_t]
#  name [char[name_size]]
#  nb_inputs [int32_t]
#  intput_ids [int32_t[nb_inputs]]
#
# (additional information)
#  Const_layer: 
#   length_dim [int32_t]
#   dim [int32_t[length_dim]]
#   type [int32_t] 0:int32, 1:float32 2:int16 
#   [if integer: quantizer [int32])
#   data [type[prod(dim)]]     
#
#  Conv2D
#    nb_dim_strides [int32_t]
#    strides [int32_t[nb_dim_strides]]
#    quantizer [int32_t]
#
#  MatMul
#    quantizer [int32_t]
#
#  Mul
#    quantizer [int32_t]
#
#  PlaceHolder
#   length_dim [int32_t]
#   dim [int32_t[length_dim]]
#   quantizer [int32_t]
#
#  MaxPool
#    nb_dim_strides [int32_t]
#    strides [int32_t[nb_dim_strides]]
#    nb_dim_kernel [int32_t]
#    kernel_dim [int32_t[nb_dim_kernel]]

def dumpModel(model, user_inputs, output_filename, verbose):
    """Writes the neural network model in the \"tf2cpp\" format to binary file.
    
    Parameters
    ----------
    model : tensorflow.python.keras.engine.functional.Functional
        Neural network model.
    output_filename : either str or None
        Path to the binary file to which the neural network model
        is written. 
    verbose : bool
        Is additional information printed?
    """

    prms = json.loads(model.to_json())
    if verbose:
       print(json.dumps(prms, indent=2))
    if output_filename is None:
    	raise ValueError('Need an output filename')
    	
    id = 0
    id_names = dict()
    layer_name = {}
    layer_op = {}
    input_dimensions = {}
    layer_padding = {}
    layer_stride = {}
    layer_op = {}
    layer_data = {}
    layer_inputs = {}
    layer_alias = {}
    for L in prms['config']['layers']:
        name = L['name']
        layer_name[id] = name
        id_names[name] = id
        layer_alias[name] = name # use when we replace some layers by another one (eg conv2D)
        layer_inputs[id] = []
        if len(L['inbound_nodes']) > 0:
            for i in L['inbound_nodes'][0]: # warning: chnage in TF 2.4: not a list of inputs anymore... 
                layer_inputs[id].append(i[0])
        
        # Specialization per class.
        if L['class_name'] == 'Conv2D': # generate 4 layers: conv2D, weights, bias, biasadd, activation
            org_name = name
            id_names[org_name + '_w'] = id
            layer_name[id] = org_name + '_w'
            layer_alias[layer_name[id]] = layer_name[id]
            layer_inputs[id + 1] = layer_inputs[id]  # save inputs
            layer_inputs[id + 1].append(org_name + '_w') # conv2d has 2 inputs: data and kernels
            layer_inputs[id] = []
            layer_op[id] = 'Const'
            layer_data[id] = model.get_layer(org_name).get_weights()[0]

            
            # Rewrite the conv2D.
            id = id + 1
            name = org_name + '_c'
            id_names[name] = id
            layer_name[id] = name
            layer_alias[layer_name[id]] = layer_name[id]
            layer_alias[org_name] = layer_name[id]
            layer_padding[id] = L['config']['padding']
            layer_stride[id] = L['config']['strides']
            layer_op[id] = 'Conv2D'
            id_conv = id
            
            # Bias conv2D.
            if L['config']['use_bias']:
                id = id + 1
                id_names[org_name + '_b'] = id
                layer_name[id] = org_name+'_b'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id] = []
                layer_op[id] = 'Const'
                layer_data[id] = model.get_layer(org_name).get_weights()[1] # layer_data[id]=model.get_layer('conv2d')

                id = id + 1
                id_names[org_name + '_badd'] = id # biasadd has 2 inputs: output of conv2d and bias
                layer_name[id] = org_name + '_badd'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id] = [name, org_name + '_b']
                layer_op[id] = 'Add'
                layer_alias[org_name] = layer_name[id]
            
            # activation conv2D
            if L['config']['activation'] == 'relu':
                id = id + 1
                id_names[org_name + '_a'] = id
                layer_name[id] = org_name + '_a'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id] = [layer_alias[org_name]]
                layer_op[id] = 'Relu'
                layer_alias[org_name] = layer_name[id]
            elif L['config']['activation'] == 'linear':
                pass
            else:
                print('TODO activation', L['config']['activation'])
                quit()
        
        elif L['class_name'] == 'Dense': # generate 4 layers: conv2D, weights, bias, biasadd, activation
            org_name = name
            
            # Weights dense replace dense layer by weights because order matters in cpp parts TO SOLVE TODO.
            id_names[org_name + '_w'] = id
            layer_name[id] = org_name + '_w'
            layer_alias[layer_name[id]] = layer_name[id]
            layer_inputs[id + 1] = layer_inputs[id] # save inputs
            layer_inputs[id + 1].append(org_name + '_w') # matmul has 2 inputs
            layer_inputs[id] = []
            layer_op[id] = 'Const'
            layer_data[id] = model.get_layer(org_name).get_weights()[0] # layer_data[id]=model.get_layer('conv2d')

            
            # Rewrite the dense.
            id = id + 1
            name = org_name + '_m' # rename the layer
            id_names[name] = id        
            layer_name[id] = name
            layer_alias[layer_name[id]] = layer_name[id]
            layer_alias[org_name] = layer_name[id]
            layer_op[id] = 'MatMul'
            id_conv = id
            
            # Bias dense.
            if L['config']['use_bias']:
                id = id + 1
                id_names[org_name + '_b'] = id
                layer_name[id] = org_name + '_b'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id] = []
                layer_op[id] = 'Const'
                layer_data[id] = model.get_layer(org_name).get_weights()[1] # layer_data[id]=model.get_layer('conv2d')

                
                id = id + 1
                id_names[org_name + '_badd'] = id # biasadd has 2 inputs: output of conv2d and bias
                layer_name[id] = org_name + '_badd'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id] = [name, org_name + '_b']
                layer_op[id] = 'Add'
                layer_alias[org_name] = layer_name[id] # replace output
            
            if L['config']['activation'] == 'relu':
                id = id + 1
                id_names[org_name + '_a'] = id
                layer_name[id] = org_name + '_a'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id] = [layer_alias[org_name]]
                layer_op[id] = 'Relu'
                layer_alias[org_name] = layer_name[id] # replace output
            elif L['config']['activation'] == 'linear':
                pass
            else:
                 print('TODO activation', L['config']['activation'])
                 quit()
        
        elif L['class_name'] == 'InputLayer':
            layer_op[id] ='Placeholder'
            input_dimensions[id] = [1 if x is None else x for x in L['config']['batch_input_shape']]

        elif L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] == 'Identity':
            layer_op[id]='Identity'
        
        elif L['class_name'] == 'Add' or L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] in ('Add', 'AddV2', 'BiasAdd'):
            
            # TODO: check what happens when two variables are involved
            # in the addition and no graph node is involved.
            # The condition below is true if a variable is fed into
            # the addition operator.
            if L['class_name'] == 'TensorFlowOpLayer':
                if len(L['config']['constants']):
                    org_name = name
                    
                    # `id` is the ID of the variable whereas `id` + 1
                    # is the ID of the addition operator.
                    id_names[org_name + '_b'] = id
                    layer_name[id] = org_name + '_b'
                    layer_alias[layer_name[id]] = layer_name[id]
                    layer_inputs[id + 1] = layer_inputs[id]
                    layer_inputs[id + 1].append(org_name + '_b')
                    layer_inputs[id] = []
                    layer_op[id] = 'Const'
                    layer_data[id] = np.array(L['config']['constants']['1'], dtype=np.float32)

                    id = id  + 1 
                    id_names[org_name] = id
                    layer_name[id] = org_name
                    layer_alias[layer_name[id]] = layer_name[id]
            layer_op[id] = 'Add'
        
        elif L['class_name'] == 'Multiply' or L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] == 'Mul':
            
            # TODO: check what happens when two variables are involved
            # in the addition and no graph node is involved.
            # The condition below is true if a variable is fed into
            # the addition operator.
            if L['class_name'] == 'TensorFlowOpLayer':
                if len(L['config']['constants']):
                    org_name = name
                    
                    # `id` is the ID of the variable whereas `id` + 1
                    # is the ID of the addition operator.
                    id_names[org_name + '_w'] = id
                    layer_name[id] = org_name + '_w'
                    layer_alias[layer_name[id]] = layer_name[id]
                    layer_inputs[id + 1] = layer_inputs[id]
                    layer_inputs[id + 1].append(org_name + '_w')
                    layer_inputs[id] = []
                    layer_op[id] = 'Const'
                    layer_data[id] = np.array(L['config']['constants']['1'], dtype=np.float32)

                    id = id  + 1 
                    id_names[org_name] = id
                    layer_name[id] = org_name
                    layer_alias[layer_name[id]] = layer_name[id]
            layer_op[id] = 'Mul'
        
        elif L['class_name'] == 'Maximum':
            layer_op[id] = 'Maximum'
        
        elif L['class_name'] == 'ReLU':
            layer_op[id] = 'Relu'
            if L['config']['max_value'] is not None or L['config']['threshold'] != 0.0 or L['config']['negative_slope'] != 0.0:
                print('TODO relu with prm')
                quit()
        
        elif L['class_name'] == 'LeakyReLU':
            org_name = name
            id_names[org_name + '_alpha'] = id
            layer_name[id] = org_name + '_alpha'
            layer_alias[layer_name[id]] = layer_name[id]
            layer_inputs[id + 1] = layer_inputs[id]
            layer_inputs[id + 1].append(org_name + '_alpha')
            layer_inputs[id] = []
            layer_op[id] = 'Const'
            layer_data[id] = np.array([model.get_layer(name).alpha.item()], dtype=np.float32)


            id  = id + 1
            layer_op[id] = 'LeakyReLU'
            id_names[name] = id
            layer_name[id] = name
        
        elif L['class_name'] == 'Concatenate' or L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] == 'ConcatV2':
            org_name = name
            id_names[org_name + '_axis'] = id
            layer_name[id] = org_name + '_axis'
            layer_alias[layer_name[id]] = layer_name[id]
            layer_inputs[id + 1] = layer_inputs[id]
            layer_inputs[id + 1].append(org_name + '_axis')
            layer_inputs[id] = []
            layer_op[id] = 'Const'
            if L['class_name'] == 'TensorFlowOpLayer':
                nb_inputs = len(layer_inputs[id + 1]) - 1
                ax = int(L['config']['constants'][str(nb_inputs)])
            else:
                ax = int(L['config']['axis'])
            layer_data[id] = np.array([ax] , dtype='int32') 
            
            id = id  + 1
            layer_op[id] = 'ConcatV2'
            id_names[org_name] = id
            layer_name[id] = org_name
            layer_alias[layer_name[id]] = layer_name[id]
        
        elif L['class_name'] == 'MaxPool' or L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] == 'MaxPool':
            layer_op[id] = 'MaxPool'
            layer_stride[id] = L['config']['node_def']['attr']['strides']['list']['i']
            
            # As the maxpoolig layer has no padding,
            # the kernel sizes are stored in the place
            # dedicated to the padding.
            layer_padding[id] = L['config']['node_def']['attr']['ksize']['list']['i']
        
        elif L['class_name'] == 'Reshape' or L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] == 'Reshape':
            org_name=name
            id_names[org_name + '_shape'] = id
            layer_name[id] = org_name + '_shape'
            layer_alias[layer_name[id]] = layer_name[id]
            layer_inputs[id + 1] = layer_inputs[id]
            layer_inputs[id + 1].append(org_name + '_shape')    
            layer_inputs[id] = [] 
            layer_op[id] = 'Const'
            if L['class_name'] == 'TensorFlowOpLayer':
                shape = L['config']['constants']['1']
            else:
                shape = L['config']['target_shape']
                shape.insert(0, -1)
            layer_data[id] = np.asarray(shape, dtype='int32')
            
            id = id + 1
            layer_op[id] = 'Reshape'
            id_names[org_name] = id 
            layer_name[id] = org_name
            layer_alias[layer_name[id]] = layer_name[id]
        
        elif L['class_name'] == 'TensorFlowOpLayer' and L['config']['node_def']['op'] in ('MatMul', 'BatchMatMulV2'):
            
            # TODO: check what happens when two variables are involved
            # in the batch matrix multiplication and no graph node
            # is involved.
            # The condition below is true if a variable is fed into
            # the addition operator.
            if len(L['config']['constants']):
                org_name = name
                
                # `id` is the ID of the variable whereas `id` + 1
                # is the ID of the batch matrix multiplication operator.
                id_names[org_name + '_w'] = id
                layer_name[id] = org_name + '_w'
                layer_alias[layer_name[id]] = layer_name[id]
                layer_inputs[id + 1] = layer_inputs[id]
                layer_inputs[id + 1].append(org_name + '_w')
                layer_inputs[id] = []
                layer_op[id] = 'Const'
                # `type(L['config']['constants']['1'])` returns `list`.
                layer_data[id] = np.array(L['config']['constants']['1'], dtype=np.float32)
                    

                id = id  + 1
                id_names[org_name] = id
                layer_name[id] = org_name
                layer_alias[layer_name[id]] = layer_name[id]
            layer_op[id] = 'MatMul'
        
        else:
            print('TODO layer ' + L['class_name'])
            print('   op: ' + L['config']['node_def']['op'])
            quit()
        id = id + 1
    
    inputs = []
    for i in range(len(prms['config']['input_layers'])):
      name= layer_alias[prms['config']['input_layers'][i][0]]
      inputs.append(id_names[name])
    outputs = []
    for i in range(len(prms['config']['output_layers'])):
      name= layer_alias[prms['config']['output_layers'][i][0]]
      outputs.append(id_names[name])
    
    # Dumping.
    fd = open(output_filename, 'wb')
    fd.write(str.encode('SADL0001'))
    # output of the network type 0: int32 1: float 2: int16 default: float(1)
    fd.write(struct.pack('i', int(1)))
    if verbose: print('# Nb layers: ', len(layer_name))
    fd.write(struct.pack('i', int(len(layer_name))))
    if verbose: print('# Nb inputs: ', len(inputs))
    fd.write(struct.pack('i', int(len(inputs))))
    for i in inputs:
      if verbose: print('#  input ', i)
      fd.write(struct.pack('i', int(i)))
    if verbose: print('# Nb outputs: ', len(outputs))
    fd.write(struct.pack('i', int(len(outputs))))
    for i in outputs:
      if verbose: print('#  output ', i)
      fd.write(struct.pack('i', int(i)))
    print('')
    cpt_user_inputs = 0
    for id in range(len(layer_name)):
        if verbose: print("# Layer id ", id )
        fd.write(struct.pack('i', int(id)))
        
        if verbose: print("#  op ", layer_op[id] )
        fd.write(struct.pack('i', int(csts.DICT_FIELD_NUMBERS[layer_op[id]])))
        
        if verbose: print("#  name_size ", len(layer_name[id]) )
        fd.write(struct.pack('i', int(len(layer_name[id]))))
        
        if verbose: print("#  name ", layer_name[id] )
        fd.write(str.encode(str(layer_name[id])))
        
        if verbose: print("#  nb_inputs ", len(layer_inputs[id]) )
        fd.write(struct.pack('i', int(len(layer_inputs[id]) )))
        
        for i in layer_inputs[id]:
          if verbose: print("#    ", id_names[layer_alias[i]], "("+layer_alias[i]+")" )
          fd.write(struct.pack('i', int(id_names[layer_alias[i]])))
        
        # custom data
        if layer_op[id] == 'Const':
           if verbose: print("#  nb_dim", len(layer_data[id].shape)) # dim [int32_t]
           fd.write(struct.pack('i', int(len(layer_data[id].shape))))
          
           for i in layer_data[id].shape:
              if verbose: print("#   ", i )
              fd.write(struct.pack('i', int(i)))
           
           data_type = -1
           if layer_data[id].dtype == 'float32':
                data_type = 1
           elif layer_data[id].dtype == 'int32':
                data_type = 0
           elif  layer_data[id].dtype == 'int16':
                data_type = 2
           elif layer_data[id].dtype == 'int8':
                data_type = 3
           else:
                print("unkwown type")
                quit()
           if verbose: print("#  dtype", data_type) # dim [int32_t]
           fd.write(struct.pack('i', int(data_type)))
           if data_type != 1: # not float
                if verbose: print("#  quantizer", int(0))
                fd.write(struct.pack('i', int(0)))
           # print("#  data: ", layer_data[id][0])
           fd.write(layer_data[id].tobytes())
           
        if layer_op[id] == 'Conv2D':
           if verbose: print("#  nb_dim_strides", len(layer_stride[id])) # dim [int32_t]
           fd.write(struct.pack('i', int(len(layer_stride[id]))))
          
           for i in layer_stride[id]:
              if verbose: print("#   ", i )
              fd.write(struct.pack('i', int(i)))

        if layer_op[id] == 'MaxPool':
           if verbose: print("#  nb_dim_strides", len(layer_stride[id])) # dim [int32_t]
           fd.write(struct.pack('i', int(len(layer_stride[id]))))
          
           for i in layer_stride[id]:
              if verbose: print("#   ", i )
              fd.write(struct.pack('i', int(i)))
           if verbose: print("# nb_dim_kernels", len(layer_padding[id]))
           fd.write(struct.pack('i', int(len(layer_padding[id]))))
           
           for i in layer_padding[id]:
             if verbose: print("#   ", i)
             fd.write(struct.pack('i', int(i)))

        if layer_op[id] in ("Conv2D", "Mul", "MatMul"):
            # output the internal quantizer default: 0
            fd.write(struct.pack('i', int(0)))

        if layer_op[id] == 'Placeholder':
            if verbose: print("#  nb input dimensions", len(input_dimensions[id]))
            fd.write(struct.pack('i', int(len(input_dimensions[id]))))
            if user_inputs[cpt_user_inputs].shape != tuple(input_dimensions[id]):
            	print("[WARN] input {}: input size {} overwrite by user input size {}".format(id,input_dimensions[id],user_inputs[cpt_user_inputs].shape))
            	for i in user_inputs[cpt_user_inputs].shape:
                  if verbose: print("#    ", i)
                  fd.write(struct.pack('i', int(i)))
            else:            
                for i in input_dimensions[id]:
                  if verbose: print("#    ", i)
                  fd.write(struct.pack('i', int(i)))
 
            cpt_user_inputs = cpt_user_inputs + 1	
            # output the quantizer of the input default: 0
            if verbose: print("#   quantizer_of_input", 0)
            fd.write(struct.pack('i', int(0)))
        
        if verbose: print("")
        
    if verbose: print("TODO: check data order")
    
    print("[INFO] dumped model in "+output_filename)
      
