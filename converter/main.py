"""
/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2022, ITU/ISO/IEC
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
import argparse
import onnx
import copy
import struct
from collections import OrderedDict
from enum import IntEnum
import numpy as np

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


class OPTYPE(IntEnum):
    Const=1,
    Placeholder=2,
    Identity=3,
    BiasAdd=4,
    MaxPool=5,
    MatMul=6,
    Reshape=7,
    Relu=8,
    Conv2D=9,
    Add=10,
    ConcatV2=11,
    Mul=12,
    Maximum=13,
    LeakyReLU=14,
    Transpose=15,
    Flatten=16,
    # In "tf2cpp", the same layer performs the matrix multiplication
    # and the matrix multiplication by batches.
    BatchMatMul = 6,
    
    # "BatchMatMulV2" did not exist in Tensorflow 1.9. It exists in
    # Tensorflow 1.15.
    BatchMatMulV2 = 6

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

class DTYPE_SADL(IntEnum):
    FLOAT = 1,   # float
    INT8 =  3,   # int8_t
    INT16 = 2,   # int16_t
    INT32 = 0    # int32_t

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name
            
class DTYPE_ONNX(IntEnum):
    # https://github.com/onnx/onnx/blob/master/onnx/onnx.in.proto#L483-L485
    FLOAT = 1,   # float
    INT8 =  3,   # int8_t
    INT16 = 4,   # int16_t
    INT32 = 6,   # int32_t
    INT64 = 7    # int64_t

    def __repr__(self):
        return self.name
    
    def __str__(self):
        return self.name

class Node_Annotation:
    to_remove=False
    add_transpose_before=False
 #   data_layout_sadl=None
    to_transpose = False
    layout_onnx = None
    
    def __repr__(self):
      return "to_remove={}, to_transpose={}, layout_onnx={}, add_transpose_before={}".format(self.to_remove,self.to_transpose,self.layout_onnx,self.add_transpose_before)

# get attribute name in node
def getAttribute(node, attr):
    for a in node.attribute:
        if a.name == attr: return a
    return None


def transpose_tensor(raw_data,dims):
    """
        When convert TF2 to ONNX, ONNX weight's  are not represent in the same way as TF2 weight's
    """
    # print(dims)
    tmp = []
    tmp.append(dims[2])
    tmp.append(dims[3])
    tmp.append(dims[1])
    tmp.append(dims[0])
    
    x = np.frombuffer(raw_data, dtype=np.float32)
    x = x.reshape(tmp[3], tmp[2], tmp[0] * tmp[1]).transpose().flatten()
    return x.tobytes(), tmp

def transpose_matrix(raw_data,dims):
    x = np.frombuffer(raw_data, dtype=np.float32)
    tmp = []
    tmp.append(dims[1])
    tmp.append(dims[0])
    x = x.reshape(dims[0],dims[1])
    x = np.transpose(x) # moveaxis(x, -2, -1) 
    return x.flatten().tobytes(), tmp

def toList(ii):
    d = []
    for i in ii: d.append(i)
    return d

def is_constant(name,onnx_initializer):
    for n in onnx_initializer:
        if n.name == name: return True
    return False

def is_output(name, onnx_output):
    for out in onnx_output:
        if out.name == name:
            return True
    return False

def parse_graph_input_node(input_node, map_onnx_to_myGraph, to_transpose):
    map_onnx_to_myGraph[input_node.name] = input_node.name
    struct = {}
    struct["inputs"] = []
    struct["additional"] = {}
    if to_transpose: # data_layout == 'nchw' and len(input_node.type.tensor_type.shape.dim)==4:
        struct["additional"]["dims"] = [input_node.type.tensor_type.shape.dim[0].dim_value, 
                                        input_node.type.tensor_type.shape.dim[2].dim_value, input_node.type.tensor_type.shape.dim[3].dim_value, input_node.type.tensor_type.shape.dim[1].dim_value]
    else:
        struct["additional"]["dims"] = [d.dim_value for d in input_node.type.tensor_type.shape.dim]
    struct["op_type"] = OPTYPE.Placeholder
    return struct

def extract_additional_data_from_node(data,to_transpose):
    tmp = {}
    if data.dims == []:
      tmp["dims"] =  [1]
    else:
      tmp["dims"] =  [dim for dim in data.dims]
          
    tmp["raw_data"] = data.raw_data

    if data.data_type == DTYPE_ONNX.FLOAT:
        tmp["dtype"] = DTYPE_SADL.FLOAT
    elif data.data_type == DTYPE_ONNX.INT8:
        tmp["dtype"] = DTYPE_SADL.INT8
    elif data.data_type == DTYPE_ONNX.INT16:
        tmp["dtype"] = DTYPE_SADL.INT16
    elif data.data_type == DTYPE_ONNX.INT32:
        tmp["dtype"] = DTYPE_SADL.INT32
    elif data.data_type == DTYPE_ONNX.INT64:
        def convert_int64_to_int32(binary_data):
            x = np.frombuffer(binary_data, dtype=np.int64)
            x = x.astype(np.int32)
            return x.tobytes()
        tmp["dtype"] = DTYPE_SADL.INT32
        tmp["raw_data"] = convert_int64_to_int32(tmp["raw_data"])
    else:
        raise ValueError("extract_additional_data: Unknown dtype")
        
    if to_transpose:
      if   len(tmp["dims"]) == 4:
           tmp["raw_data"], tmp["dims"] = transpose_tensor(tmp["raw_data"], tmp["dims"]) 
      elif len(tmp["dims"]) == 2: #  and data_layout == "nchw":
           tmp["raw_data"], tmp["dims"]  = transpose_matrix(tmp["raw_data"], tmp["dims"])
    
    return tmp["dims"], tmp["raw_data"], tmp["dtype"]    
    
def extract_additional_data(name, to_transpose, onnx_graph):
    for init in onnx_graph.initializer:
        if name == init.name:      return extract_additional_data_from_node(init,to_transpose)
    for node in onnx_graph.node: # not found in initializaer, search in Constant
        if name == node.output[0]: return extract_additional_data_from_node(node.attribute[0].t, to_transpose)
    quit("[ERROR] unable to extract data in {}".format(name))
    
def extract_dims(name, onnx_graph):
    for init in onnx_graph.initializer:
        if name == init.name:      return init.dims
    for node in onnx_graph.node: # not found in initializaer, search in Constant
        if name == node.output[0]: 
            a = getAttribute(node,"value")
            if a is not None: 
                return a.t.dims
            else:
                return  None
    for node in onnx_graph.input: # not found in initializaer, search in Constant
        if name == node.name: return node.type.tensor_type.shape.dim
    quit("[ERROR] unable to extract dims in {}".format(name))
    
    
# get the nodes with name as input
def getNodesWithInput(name,model):
    L= []
    for node in model.graph.node:
         for inp in node.input:
             if inp == name: 
                 L.append(node)
    return L   
    
# get the nodes with name as output
def getNodesWithOutput(name,model):
    for node in model.graph.node:
         for out in node.output:
             if out == name: 
                 return node             
    for node in model.graph.initializer:
             if node.name == name: 
                 return node
    for node in model.graph.input:
             if node.name == name: 
                 return node
    quit("[ERROR] not found:".format(name))

# get the nodes with name as output
def getNodesWithOutputNotConst(name,model):
    for node in model.graph.node:
         for out in node.output:
             if out == name: 
                 return node             
    for node in model.graph.input:
             if node.name == name: 
                 return node
    return None
        
# get dims from data
def getDims(node):
    if node.data_type != DTYPE_ONNX.INT64: 
        quit("[ERROR] bad node type fpr getDims {}".format(node))
        
    x = np.frombuffer(node.raw_data, dtype=np.int64)
    dims = x.tolist()
    return dims
    
def getInitializer(name,model_onnx):
    for node in model_onnx.graph.initializer:
        if node.name == name: return node
    return None
    
def add_transpose(node,myGraph,map_onnx_to_myGraph):
     # Transpose inserted
    # Const
    reshape_coef_name = node.input[0]+ "_COEF_TRANSPOSE_NOT_IN_GRAPH"
    myGraph[reshape_coef_name] = {}
    myGraph[reshape_coef_name]["op_type"] = OPTYPE.Const 
    myGraph[reshape_coef_name]["inputs"] = []         
    additional = {}                    
    additional["dims"] = [4]
    additional["raw_data"] = np.array([0,3,1,2], dtype=np.int32).tobytes()
    additional["dtype"] = DTYPE_SADL.INT32
    additional["data"] = node 
    myGraph[reshape_coef_name]["additional"] = additional
    map_onnx_to_myGraph[reshape_coef_name] = reshape_coef_name
    
    nname = node.input[0]+ "_TRANSPOSE_NOT_IN_GRAPH"
    myGraph[nname] = {}
    myGraph[nname]["op_type"] = OPTYPE.Transpose
    myGraph[nname]["inputs"] = [map_onnx_to_myGraph[node.input[0]], reshape_coef_name]         
    map_onnx_to_myGraph[nname] = nname
    return nname   
    
def parse_graph_node(node, model_onnx, myGraph, node_annotation, map_onnx_to_myGraph,verbose):
    if verbose>1: print("parse node",node.name)
    
    if node_annotation[node.name].add_transpose_before: # layout_onnx == 'nchw' : # need to go back to original layout before reshape
       n0name = add_transpose(node,myGraph,map_onnx_to_myGraph)           
    else:
       if len(node.input)>=1: n0name = node.input[0]
       else:                  n0name=None


    if node.op_type == "Conv" or node.op_type == "Gemm":
        nb_inputs = len(node.input)        
        if (nb_inputs != 3) and (nb_inputs != 2 ): raise Exception("parse_graph_node: Error on node type")                   
        additional = {}                    
        # Const: weight
        additional["data"] = node
        n2=getNodesWithOutput(node.input[1],model_onnx)
        additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(node.input[1], node_annotation[n2.name].to_transpose, model_onnx.graph)
        map_onnx_to_myGraph[node.input[1]] = node.input[1]
        
        myGraph[node.input[1]] = {}
        myGraph[node.input[1]]["inputs"] = []
        myGraph[node.input[1]]["additional"] = additional
        myGraph[node.input[1]]["op_type"] = OPTYPE.Const
        
        # Conv2d
        inputs, additional = [], {}
        inputs = [map_onnx_to_myGraph[n0name]] + [map_onnx_to_myGraph[node.input[1]]]
        
        additional["data"] = node
        if node.op_type == "Conv":
          a = getAttribute(node,'strides')
          additional["strides"] = a.ints
          
        if nb_inputs == 2:
            map_onnx_to_myGraph[node.output[0]] = node.output[0]
        elif nb_inputs == 3:
            map_onnx_to_myGraph[node.output[0]] = node.output[0] + "_NOT_IN_GRAPH"
                  
        myGraph[node.output[0]] = {}
        myGraph[node.output[0]]["inputs"] = inputs
        myGraph[node.output[0]]["additional"] = additional
        if node.op_type == "Conv":
            myGraph[node.output[0]]["op_type"] = OPTYPE.Conv2D
        elif node.op_type == "Gemm":
            myGraph[node.output[0]]["op_type"] = OPTYPE.MatMul
                
        if nb_inputs == 3:
            additional = {}                    
            # Const: bias
            additional["data"] = node          
            additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(node.input[2], False, model_onnx.graph)
            map_onnx_to_myGraph[node.input[2]] = node.input[2]          
            myGraph[node.input[2]] = {}
            myGraph[node.input[2]]["inputs"] = []
            myGraph[node.input[2]]["additional"] = additional
            myGraph[node.input[2]]["op_type"] = OPTYPE.Const                    
            # BiasAdd
            inputs, additional = [], {}
            inputs = [node.output[0]] + [map_onnx_to_myGraph[node.input[2]]]
            additional["data"] = node
            map_onnx_to_myGraph[node.output[0] + "_NOT_IN_GRAPH"] = None                    
            myGraph[node.output[0] + "_NOT_IN_GRAPH"] = {}
            myGraph[node.output[0] + "_NOT_IN_GRAPH"]["inputs"] = inputs
            myGraph[node.output[0] + "_NOT_IN_GRAPH"]["additional"] = additional
            myGraph[node.output[0] + "_NOT_IN_GRAPH"]["op_type"] = OPTYPE.BiasAdd
       
    elif node.op_type == "Relu":
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Relu
        myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name]]
        myGraph[node.output[0]]["additional"] = {} 
        myGraph[node.output[0]]["additional"]["data"] = node                          
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
        
    elif node.op_type == "Constant": # ~ like an initializer
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Const
        myGraph[node.output[0]]["inputs"] = []
        myGraph[node.output[0]]["additional"] = {} 
        myGraph[node.output[0]]["additional"]["data"] = node                          
        map_onnx_to_myGraph[node.output[0]] = node.output[0]

    elif node.op_type == "Add":
        swap_inputs = False
        if is_constant(n0name,model_onnx.graph.initializer):
            additional = {}
            additional["data"] = node          
            additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(n0name, False, model_onnx.graph)
            map_onnx_to_myGraph[n0name] = n0name               
            myGraph[n0name] = {}
            myGraph[n0name]["inputs"] = []
            myGraph[n0name]["additional"] = additional
            myGraph[n0name]["op_type"] = OPTYPE.Const
            swap_inputs = True
        if is_constant(node.input[1],model_onnx.graph.initializer):
            additional = {}
            additional["data"] = node          
            additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(node.input[1], False, model_onnx.graph)
            map_onnx_to_myGraph[node.input[1]] = node.input[1]            
            myGraph[node.input[1]] = {}
            myGraph[node.input[1]]["inputs"] = []
            myGraph[node.input[1]]["additional"] = additional
            myGraph[node.input[1]]["op_type"] = OPTYPE.Const    
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Add
        if not swap_inputs:
            D1=extract_dims(n0name, model_onnx.graph)
            D2=extract_dims(node.input[1], model_onnx.graph)       
            if D1 is not None and D2 is not None and len(D1)<len(D2): swap_inputs = True
            
        if swap_inputs: myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[node.input[1]],map_onnx_to_myGraph[n0name]]
        else:           myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name],map_onnx_to_myGraph[node.input[1]]]
        myGraph[node.output[0]]["additional"] = {} 
        myGraph[node.output[0]]["additional"]["data"] = node                          
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
    
    elif node.op_type == "MaxPool":
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.MaxPool
        myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name]]
        myGraph[node.output[0]]["additional"] = {}        
        a = getAttribute(node,'strides')
        myGraph[node.output[0]]["additional"]["strides"] = [1, a.ints[0], a.ints[1], 1]
        a = getAttribute(node,'kernel_shape')
        myGraph[node.output[0]]["additional"]["kernel_shape"] = [1, a.ints[0], a.ints[1], 1]
        myGraph[node.output[0]]["additional"]["data"] = node  
        # todo: check pads?                        
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
    
    elif node.op_type == "Mul":
        # check the inputs
        if is_constant(n0name,model_onnx.graph.initializer) and is_constant(node.input[1],model_onnx.graph.initializer):
            quit("[ERROR] unsupported double constants Mul",node)
        swap_inputs = False
        if is_constant(n0name,model_onnx.graph.initializer):
            additional = {}
            additional["data"] = node          
            n2=getNodesWithOutput(n0name,model_onnx)
            additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(n0name, node_annotation[n2.name].to_transpose, model_onnx.graph)
            map_onnx_to_myGraph[n0name] = n0name               
            myGraph[n0name] = {}
            myGraph[n0name]["inputs"] = []
            myGraph[n0name]["additional"] = additional
            myGraph[n0name]["op_type"] = OPTYPE.Const
            swap_inputs = True
        if is_constant(node.input[1],model_onnx.graph.initializer):
            additional = {}
            additional["data"] = node          
            n2=getNodesWithOutput(node.input[1],model_onnx)
            additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(node.input[1], node_annotation[n2.name].to_transpose, model_onnx.graph)
            map_onnx_to_myGraph[node.input[1]] = node.input[1]            
            myGraph[node.input[1]] = {}
            myGraph[node.input[1]]["inputs"] = []
            myGraph[node.input[1]]["additional"] = additional
            myGraph[node.input[1]]["op_type"] = OPTYPE.Const    
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Mul
        if swap_inputs: myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[node.input[1]],map_onnx_to_myGraph[n0name]]
        else:           myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name],map_onnx_to_myGraph[node.input[1]]]
        myGraph[node.output[0]]["additional"] = {} 
        myGraph[node.output[0]]["additional"]["data"] = node                          
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
    
    elif node.op_type == "Identity":
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Identity
        myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name]]
        myGraph[node.output[0]]["additional"] = {} 
        myGraph[node.output[0]]["additional"]["data"] = node                          
        map_onnx_to_myGraph[node.output[0]] = node.output[0]   
    
    elif node.op_type == "LeakyRelu":
        # leaky coef
        additional = {}
        additional["data"] = node
        additional["dims"] = [1]
        additional["raw_data"] =  np.array(float(node.attribute[0].f),dtype=np.float32).tobytes()
        additional["dtype"] = DTYPE_SADL.FLOAT
        map_onnx_to_myGraph[node.output[0] + "_COEF_NOT_IN_GRAPH"] = None                    
        myGraph[node.output[0] + "_NOT_IN_GRAPH"] = {}
        myGraph[node.output[0] + "_NOT_IN_GRAPH"]["inputs"] = []
        myGraph[node.output[0] + "_NOT_IN_GRAPH"]["additional"] = additional
        myGraph[node.output[0] + "_NOT_IN_GRAPH"]["op_type"] = OPTYPE.Const

        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.LeakyReLU
        myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name],node.output[0] + "_NOT_IN_GRAPH"]
        myGraph[node.output[0]]["additional"] = {} 
        myGraph[node.output[0]]["additional"]["data"] = node                          
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
                
    elif node.op_type == "Flatten":        
        inputs, additional = [], {}
        inputs = [map_onnx_to_myGraph[n0name]]
        additional["data"] = node
        a = getAttribute(node,"axis")
        additional["axis"] = a.i
        myGraph[node.output[0]] = {}
        myGraph[node.output[0]]["inputs"] = inputs
        myGraph[node.output[0]]["additional"] = additional
        myGraph[node.output[0]]["op_type"] = OPTYPE.Flatten
        map_onnx_to_myGraph[node.output[0]] = node.output[0]



    elif node.op_type == "Reshape" or node.op_type == "MatMul":
        # Const
        myGraph[node.input[1]] = {}
        myGraph[node.input[1]]["op_type"] = OPTYPE.Const 
        myGraph[node.input[1]]["inputs"] = []         
        additional = {}                    
        additional["dims"], additional["raw_data"], additional["dtype"] = extract_additional_data(node.input[1], False, model_onnx.graph)
        additional["data"] = node 
        myGraph[node.input[1]]["additional"] = additional
        map_onnx_to_myGraph[node.input[1]] = node.input[1]
        n2 = getNodesWithOutput(node.input[0], model_onnx)
        # Reshape
        inputs, additional = [], {}
        inputs = [map_onnx_to_myGraph[n0name],node.input[1]]
        additional["data"] = node
        myGraph[node.output[0]] = {}
        myGraph[node.output[0]]["inputs"] = inputs
        myGraph[node.output[0]]["additional"] = additional
        
        if node.op_type == "Reshape":
            myGraph[node.output[0]]["op_type"] = OPTYPE.Reshape
        elif node.op_type == "MatMul":
            myGraph[node.output[0]]["op_type"] = OPTYPE.MatMul
            
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
    
    elif node.op_type == "Concat":
        # Const
        myGraph[node.output[0]] = {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Const 
        myGraph[node.output[0]]["inputs"] = []         
        additional = {}
        additional["dims"] = [1]
        additional["raw_data"] = np.array(node.attribute[0].i, dtype=np.int32).tobytes()
        additional["dtype"] = DTYPE_SADL.INT32
        additional["data"] = node 
        myGraph[node.output[0]]["additional"] = additional
        map_onnx_to_myGraph[node.output[0]] = node.output[0] + "_NOT_IN_GRAPH"
        
        # Concatenate
        inputs, additional = [], {}
        for inp in node.input:
            inputs.append(map_onnx_to_myGraph[inp])
        inputs.append(node.output[0])
        additional["data"] = node
        myGraph[node.output[0] + "_NOT_IN_GRAPH"] = {}
        myGraph[node.output[0] + "_NOT_IN_GRAPH"]["inputs"] = inputs
        myGraph[node.output[0] + "_NOT_IN_GRAPH"]["additional"] = additional
        myGraph[node.output[0] + "_NOT_IN_GRAPH"]["op_type"] = OPTYPE.ConcatV2
            
        map_onnx_to_myGraph[node.output[0] + "_NOT_IN_GRAPH"] = None
    
    elif node.op_type == "Max":
        myGraph[node.output[0]]= {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Maximum
        myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name], map_onnx_to_myGraph[node.input[1]]]
        myGraph[node.output[0]]["additional"] = {}
        myGraph[node.output[0]]["additional"]["data"] = node 
        map_onnx_to_myGraph[node.output[0]] = node.output[0]

    elif node.op_type == "Unsqueeze":
        # No need to parse Unsqueeze as SADL can handle it.
        map_onnx_to_myGraph[node.output[0]] = node.output[0]
        
    elif node.op_type == "Transpose":
        # Const
        reshape_coef_name = node.output[0]+ "_COEF_TRANSPOSE"
        myGraph[reshape_coef_name] = {}
        myGraph[reshape_coef_name]["op_type"] = OPTYPE.Const 
        myGraph[reshape_coef_name]["inputs"] = []         
        additional = {}      
        d = toList(getAttribute(node,"perm").ints)   
        additional["dims"] = [len(d)]
        additional["raw_data"] = np.array(d, dtype=np.int32).tobytes()
        additional["dtype"] = DTYPE_SADL.INT32
        additional["data"] = node 
        myGraph[reshape_coef_name]["additional"] = additional
        map_onnx_to_myGraph[reshape_coef_name] = reshape_coef_name
        
        myGraph[node.output[0]] = {}
        myGraph[node.output[0]]["op_type"] = OPTYPE.Transpose
        myGraph[node.output[0]]["inputs"] = [map_onnx_to_myGraph[n0name], reshape_coef_name]         
        map_onnx_to_myGraph[node.output[0]] = node.output[0]

    else:
        raise Exception("[ERROR] node not supported:\n{})".format(node))

def parse_onnx(model_onnx, node_annotation, verbose=False):
    myGraph, map_onnx_to_myGraph = OrderedDict(), {}
    
    # Inputs
    for inp in model_onnx.graph.input:
        myGraph[inp.name] = parse_graph_input_node(inp, map_onnx_to_myGraph,node_annotation[inp.name].to_transpose)
    
    # Nodes removal
    for node in model_onnx.graph.node:
        if node.name in node_annotation and node_annotation[node.name].to_remove:
            curr_key = node.input[0]
            while map_onnx_to_myGraph[curr_key] != None and map_onnx_to_myGraph[curr_key] != curr_key:
                next_key = map_onnx_to_myGraph[curr_key]
                curr_key = next_key
                if curr_key not in map_onnx_to_myGraph:
                    curr_key = node.input[0]
                    break
                
            map_onnx_to_myGraph[node.output[0]] = curr_key
        else:
            parse_graph_node(node, model_onnx, myGraph,node_annotation, map_onnx_to_myGraph,verbose)
                
    myInputs = []
    for inp in model_onnx.graph.input:
        myInputs.append(inp.name)

    myOutputs = []
    for out in model_onnx.graph.output:
        for key, value in map_onnx_to_myGraph.items():
            if key == out.name:
                myOutputs.append(value)
            
    return myGraph, myInputs, myOutputs

def dump_onnx(graph, my_inputs, my_outputs,output_filename,verbose=False):
    # graph[my_name]={ op_type
    #                  inputs: [] 
    #                  dtype: 
    #                  onnx : model.graph.node[x]
    #                  }
    
    # my_input=[my_name, my_name..]
    # outputs=[my_name, ...]
    # print(graph)
    map_name_to_idx = dict()
    for idx, (key, value) in enumerate(graph.items()):
      map_name_to_idx[key] = idx
    
    # dbg print(map_name_to_idx)
    with open(output_filename, "wb") as f:
        f.write(str.encode('SADL0001'))
        # output of the network type 0: int32 | 1: float | 2: int16 | default: float(1)
        f.write(struct.pack('i', int(DTYPE_SADL.FLOAT)))
        
        if verbose: print(f"# Nb layers: {len(graph.keys())}")
        f.write(struct.pack('i', int(len(graph.keys()))))

        inputs = []
        for name in my_inputs:
            inputs.append(map_name_to_idx[name])        
        if verbose: print(f"# Nb inputs: {len(inputs)}")
        f.write(struct.pack('i', int(len(inputs))))
        for i in inputs:
            if verbose: print(f'#  input',i)
            f.write(struct.pack('i', int(i)))

        outputs = []        
        for name in my_outputs:
            outputs.append(map_name_to_idx[name])        
        if verbose: print(f"# Nb outputs: {len(outputs)}")
        f.write(struct.pack('i', int(len(outputs))))
        for i in outputs: 
            if verbose: print(f'#  output {i}')
            f.write(struct.pack('i', int(i)))

        for (name, node) in graph.items():
            if verbose: print(f"# Layer id {map_name_to_idx[name]}")
            f.write(struct.pack('i', int(map_name_to_idx[name])))

            if verbose: print("#\t op " + str(node['op_type']))
            f.write(struct.pack('i', int(node['op_type'].value)))

            # Name size
            if verbose: print(f"#\t name_size {len(name)}")
            f.write(struct.pack('i', int(len(name))))

            # Name
            if verbose: print(f"#\t name {name}")
            f.write(str.encode(str(name)))

            # Nb inputs
            if verbose: print(f"#\t nb_inputs {len(node['inputs'])}")
            f.write(struct.pack('i', int(len(node['inputs']))))

            for name_i in node['inputs']:
                idx = map_name_to_idx[name_i]
                if verbose: print(f"#\t\t {idx} ({name_i})")
                f.write(struct.pack('i', int(idx)))

            # Additional info depending on OPTYPE
            if node['op_type'] == OPTYPE.Const:
                if verbose: print(f"#\t nb_dim {len(node['additional']['dims'])}")
                f.write(struct.pack('i', int(len(node['additional']['dims']))))

                for dim in node['additional']['dims']:
                    if verbose: print(f"#\t\t {dim}")
                    f.write(struct.pack('i', int(dim)))

                if verbose: print(f"#\t dtype {node['additional']['dtype']}")
                f.write(struct.pack('i', int(node['additional']['dtype'])))

                if node['additional']['dtype'] != DTYPE_SADL.FLOAT: # not float
                    if verbose: print(f"#\t quantizer 0")
                    f.write(struct.pack('i', int(0)))

                f.write(node['additional']['raw_data'])
            # ???    if "alpha" in layer['additional']:
            #        f.write(struct.pack('f', float(layer['additional']['alpha'])))

            elif node['op_type'] == OPTYPE.Conv2D:
                if verbose: print("#\t  nb_dim_strides", len(node['additional']['strides']))
                f.write(struct.pack('i', int(len(node['additional']['strides']))))

                for stride in node['additional']['strides']:
                    if verbose: print(f"#\t\t {stride}")
                    f.write(struct.pack('i', int(stride)))

            elif node['op_type'] == OPTYPE.Placeholder:
                if verbose: print(f"#\t nb input dimension {len(node['additional']['dims'])}")
                f.write(struct.pack('i', int(len(node['additional']['dims']))))

                for dim in node['additional']['dims']:
                    if verbose: print(f"#\t\t {dim}")
                    f.write(struct.pack('i', int(dim)))

                # output the quantizer of the input default: 0
                if verbose: print(f"#\t quantizer_of_input 0")
                f.write(struct.pack('i', int(0)))   

            elif node['op_type'] == OPTYPE.MaxPool:
                if verbose: print("#\t  nb_dim_strides", len(node['additional']['strides']))
                f.write(struct.pack('i', int(len(node['additional']['strides']))))
                
                for stride in node['additional']['strides']:
                    if verbose: print(f"#\t\t {stride}")
                    f.write(struct.pack('i', int(stride)))
                
                if verbose: print("#\t  nb_dim_kernel", len(node['additional']['kernel_shape']))
                f.write(struct.pack('i', int(len(node['additional']['kernel_shape']))))
                
                for ks in node['additional']['kernel_shape']:
                    if verbose: print(f"#\t\t {ks}")
                    f.write(struct.pack('i', int(ks)))

            elif node['op_type'] == OPTYPE.Flatten:
                if verbose: print("#\t axis", node['additional']['axis'])
                f.write(struct.pack('i', int(node['additional']['axis'])))
 
            if node['op_type'] == OPTYPE.Conv2D or node['op_type'] == OPTYPE.MatMul or node['op_type'] == OPTYPE.Mul:
                # output the internal quantizer default: 0
                f.write(struct.pack('i', int(0)))

            if verbose: print("")

    
# adatp (remove/add) the current node to the data_layout and 
# recurse in the output
def annotate_node(node,model_onnx,node_annotation,global_data_layout,verbose): # recusrive
    if node.name in node_annotation: return
    if verbose>1: print("[INFO] annotate {}".format(node.name))   
    
    data_layout = None
    
    # inherit from input
    for inp in node.input:
        n2=getNodesWithOutputNotConst(inp,model_onnx)
        if n2 is not None:
            if n2.name in node_annotation:
                if data_layout is None:
                    data_layout = node_annotation[n2.name].layout_onnx
                elif node_annotation[n2.name].layout_onnx != None and node_annotation[n2.name].layout_onnx != data_layout:
                    quit("[ERROR] inputs with diferent layout for\n{}Layouts: {}".format(node,node_annotation))
            else: # not ready yet
                return
            
    if verbose>1 and data_layout is None: print("[WARNING] no data layout constraints for {}\n {}".format(node.name,node))
    
    if node.name not in node_annotation: node_annotation[node.name]=Node_Annotation()
    node_annotation[node.name].layout_onnx=data_layout # default
    
    if node.op_type == "Transpose":
        a = getAttribute(node,"perm")
        if data_layout == 'nhwc':
            if a.ints[0]==0 and a.ints[1]==3 and a.ints[2]==1 and a.ints[3]==2: # nhwc ->nchw
                node_annotation[node.name].to_remove=True # will be removed
                node_annotation[node.name].layout_onnx='nchw' # new layout at output
            else:
                if verbose>1: print("[WARNING] transpose not for NCHW handling in\n",node)
        elif data_layout == 'nchw':        
            if a.ints[0]==0 and a.ints[1]==2 and a.ints[2]==3 and a.ints[3]==1: # nchw ->nhwc
                node_annotation[node.name].to_remove=True # will be removed
                node_annotation[node.name].layout_onnx='nhwc' # new layout at output
            else:
                if verbose>1: print("[WARNING] transpose not for NCHW handling in\n",node)
    
    elif node.op_type == "Reshape":
        initializer = getInitializer(node.input[1], model_onnx)
        # Case: In pytorch, Reshape is not in model_onnx.graph.initializer but in model_onnx.graph.node
        if initializer == None:
            attribute = getAttribute(getNodesWithOutput(node.input[1], model_onnx), "value")
            initializer = attribute.t
        dims = getDims(initializer)

        # detect if this reshape is actually added by onnx to emulate a transpose
        # we need to test more if reshpae is for transpose...
        if len(dims) == 4 and ( dims[0] == 1 or  dims[0] == -1) :
            if data_layout == 'nhwc':
                if dims[1] == 1 : # or dims2 * dims3 == 1 # nhwc ->nchw
                    node_annotation[node.name].to_remove=True # will be removed
                    node_annotation[node.name].layout_onnx='nchw' # new layout at output
                else:
                    if verbose>1: print("[WARNING] reshape unknown for",node," dims",dims)
                    node_annotation[node.name].layout_onnx=None
            elif data_layout == 'ncwh':        
                if dims[3] == 1 : # # or dims2 * dims3 == 1 nchw ->nhwc
                    node_annotation[node.name].to_remove=True # will be removed
                    node_annotation[node.name].layout_onnx='nhwc' # new layout at output
                else:
                    if verbose>1: print("[WARNING] reshape unknown for",node," dims",dims)
                    node_annotation[node.name].layout_onnx=None
            elif data_layout == None:
                node_annotation[node.name].layout_onnx=global_data_layout # back to org
        else:
            node_annotation[node.name].layout_onnx=None
        
        n2 = getNodesWithOutputNotConst(node.input[0], model_onnx)
        if  node_annotation[n2.name].layout_onnx == 'nchw' : # need to go back to original layout before reshape
            node_annotation[node.name].add_transpose_before=True
    
    elif node.op_type == "Flatten":       
        if  node_annotation[node.name].layout_onnx == 'nchw' : # need to go back to original layout before reshape
            node_annotation[node.name].add_transpose_before=True
      
    elif node.op_type == 'Concat':
        if data_layout == 'nchw': # nhwc -> nhwc
           a=getAttribute(node,'axis')
           if    a.i == 1: a.i=3
           elif  a.i == 2: a.i=1
           elif  a.i == 3: a.i=2
    
    elif node.op_type == 'Unsqueeze':
        node_annotation[node.name].to_remove=True
    
    elif node.op_type == 'Conv':
           n2 = getInitializer(node.input[1],model_onnx) 
           node_annotation[n2.name].to_transpose=True
           node_annotation[n2.name].layout_onnx = 'nhwc'
    
    elif node.op_type == 'Gemm':
           n2 = getInitializer(node.input[1],model_onnx) 
           if global_data_layout == 'nchw':
               node_annotation[n2.name].to_transpose=True
           #    node_annotation[n2.name].layout_onnx = 'nhwc'
    
    nexts = getNodesWithInput(node.output[0],model_onnx)
    for n in nexts:
        annotate_node(n,model_onnx,node_annotation,global_data_layout,verbose) # rec    


def annotate_graph(model_onnx,node_annotation, data_layout,verbose):
        
    # track the data layout in the graph and remove/add layers if necessary        
    for inp in model_onnx.graph.input:
        node_annotation[inp.name]=Node_Annotation()
        if len(inp.type.tensor_type.shape.dim) == 4:
            node_annotation[inp.name].layout_onnx=data_layout   
            if data_layout == 'nchw': 
                 node_annotation[inp.name].to_transpose=True
        else:
            node_annotation[inp.name].layout_onnx=None
        
    for inp in model_onnx.graph.initializer:
        node_annotation[inp.name]=Node_Annotation()
        node_annotation[inp.name].layout_onnx=None
            
    for inp in model_onnx.graph.node:
        if inp.op_type == "Constant":
          node_annotation[inp.name]=Node_Annotation()
          node_annotation[inp.name].layout_onnx=None
            
        
    for inp in model_onnx.graph.input:     
        nexts = getNodesWithInput(inp.name, model_onnx)
        for n in nexts:
            annotate_node(n,model_onnx,node_annotation,data_layout,verbose) # recusrive
        
    if verbose > 1:
      for node in model_onnx.graph.node:
          if node.op_type == "Transpose" and (node.name not in node_annotation or not node_annotation[node.name].to_remove):
            print("[ERROR] preprocess_onnxGraph: all transpose node should be removed but this is not the case here: {}\n{}".format(node.name,node))
        
                    
                    
def detectDataType(model): # more adaptation to do here if tf is using nchw
    if model.producer_name == 'tf2onnx':
        return 'nhwc'
    elif model.producer_name == 'pytorch':
        return 'nchw'
    else:
        quit('[ERROR] unable to detect data layout')
        
    
def dumpModel(model_onnx, output_filename, data_layout, verbose):
    """Writes the neural network model in the \"sadl\" format to binary file.
    
    Parameters
    ----------
    model : onnx model
    output_filename : either str or None
        Path to the binary file to which the neural network model
        is written. 
    data_type: None, 'ncwh' or 'nwhc'
    verbose : bool
        Is additional information printed?
    """
    model_onnx_copy = copy.deepcopy(model_onnx)
    if data_layout is None: data_layout = detectDataType(model_onnx_copy)
    
    if verbose: print("[INFO] assume data type",data_layout)
    
    if verbose>1:
        # remove data
        gg = copy.deepcopy(model_onnx.graph)
        for node in gg.initializer:
            node.raw_data = np.array(0.).tobytes()
        print("[INFO] original graph:\n", gg)
        del gg
    
    
    if data_layout != 'nhwc' and data_layout != 'nchw':
        quit('[ERROR] unsupported layout', data_layout)

    node_annotation={}
    annotate_graph(model_onnx_copy, node_annotation, data_layout,verbose)

    if verbose>1: print("INFO] annotations:\n{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in node_annotation.items()) + "}") # print("[INFO] node annotations:", node_annotation)
    my_graph, my_inputs, my_outputs = parse_onnx(model_onnx_copy, node_annotation,  verbose=verbose)
    dump_onnx(my_graph, my_inputs, my_outputs,output_filename, verbose=verbose)
    if data_layout == 'nchw': print("[INFO] in SADL, your inputs and outputs has been changed from NCHW to NHWC")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='onnx2sadl conversion',
                                     usage='NB: force run on CPU')
    parser.add_argument('--input_onnx',
                        action='store',
                        nargs='?',
                        type=str,
                        help='name of the onnx file')
    parser.add_argument('--output',
                        action='store',
                        nargs='?',
                        type=str,
                        help='name of model binary file')                        
    parser.add_argument('--nchw', action='store_true')
    parser.add_argument('--nhwc', action='store_true')
    parser.add_argument('--verbose', action="count")
    args = parser.parse_args()
    if args.input_onnx is None:
        raise('[ERROR] You should specify an onnx file') 
        quit()
    if args.output is None:
        raise('[ERROR] You should specify an output file') 
        quit()
    
    print("[INFO] ONNX converter")
    if args.verbose is None: args.verbose=0
    
    model_onnx = onnx.load(args.input_onnx)
   
    data_layout = None
    if args.nchw:
      data_layout = 'nchw'
    elif args.nhwc:
      data_layout = 'nhwc'
    
    dumpModel(model_onnx, args.output, data_layout, args.verbose)
    
