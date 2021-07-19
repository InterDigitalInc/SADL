"""A library defining functions dumping models in Tensorflow 1.x.
The copyright in this software is being made available under the BSD
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

import copy
import numpy
import struct
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
import constants.constants as csts

def convert_vars_to_constants_write_graph(sess, list_names_nodes_input, list_names_nodes_output, list_nodes_input,
                                          path_to_file='', is_verbose=False):
    """Converts the variables in the graph to constants and writes the graph to the binary file.
    
    Parameters
    ----------
    sess : Session
        Session that runs the graph.
    list_names_nodes_input : list
        Names of the nodes fed into the graph.
    list_names_nodes_output : list
        Names of the nodes at the graph output.
    list_nodes_input : list
        `list_nodes_input[i]` is the input node of index i.
    path_to_file : str, optional
        Path to the binary file to which the graph
        is written. `path_to_file` ends with ".pbtxt".
        The default value is ''.
    is_verbose : bool, optional
        If True, more verbose is set. The default value
        is False.
    
    """
    graph = convert_variables_to_constants(sess,
                                           sess.graph.as_graph_def(),
                                           list_names_nodes_output)
    dict_nodes_equivalent = create_mapping_equivalent(graph)
    dict_nodes_output = create_mapping_input_outputs(graph,
                                                     dict_nodes_equivalent)
    dict_nodes_ids = create_mapping_name_id(graph,
                                            dict_nodes_equivalent,
                                            dict_nodes_output,
                                            list_names_nodes_input[0])
    dict_nodes_input_shape = create_mapping_input_shape(list_names_nodes_input,
                                                        list_nodes_input)
    if not path_to_file:
        raise ValueError('The path to the binary file is empty.')
    with open(path_to_file, 'wb') as file:
        write_graph_global_infos(list_names_nodes_input,
                                 list_names_nodes_output,
                                 dict_nodes_ids,
                                 file,
                                 is_verbose)
        
        # `list_nodes_done` stores the names of the nodes
        # whose attributes are already written to the binary
        # file.
        list_nodes_done = []
        
        def write_tree(name_node):
            if name_node in list_nodes_done:
                return
            node = next((node_search for node_search in graph.node if node_search.name == name_node),
                        None)
            list_nodes_done.append(name_node)
            for name_node_in in node.input:
                write_tree(dict_nodes_equivalent[name_node_in])
            
            # The ID of `node` is written to the binary file.
            file.write(struct.pack('i', dict_nodes_ids[node.name]))
            if is_verbose:
                print('Verbose - ID of the current node: {}'.format(dict_nodes_ids[node.name]))
            
            # The ID of `node.op` is written to the binary file.
            file.write(struct.pack('i', csts.DICT_FIELD_NUMBERS[node.op]))
            if is_verbose:
                print('Verbose - ID of the current operation: {}'.format(csts.DICT_FIELD_NUMBERS[node.op]))
            
            # The length of `node.name` is written to
            # the binary file.
            file.write(struct.pack('i', len(node.name)))
            if is_verbose:
                print('Verbose - Length of the node name: {}'.format(len(node.name)))
            
            # `node.name` is written to the binary file.
            bytes_to_be_written = node.name.encode('utf-8')
            file.write(bytes_to_be_written)
            if is_verbose:
                print('Verbose - Node name: {}'.format(bytes_to_be_written))
            
            # The number of nodes fed into `node` are written
            # to the binary file.
            file.write(struct.pack('i', len(node.input)))
            if is_verbose:
                print('Verbose - Number of input nodes: {}'.format(len(node.input)))
            
            # The ID of each node fed into `node` is written
            # to the binary file.
            for name_node_in in node.input:
                name_node_in_equi = dict_nodes_equivalent[name_node_in]
                file.write(struct.pack('i', dict_nodes_ids[name_node_in_equi]))
                if is_verbose:
                    print('Verbose - ID of the input node of name {0}: {1}'.format(name_node_in_equi, dict_nodes_ids[name_node_in_equi]))
            write_attribute(node,
                            file)
            if name_node in dict_nodes_input_shape:
                input_shape = dict_nodes_input_shape[name_node]
                
                # write number of dimensions
                file.write(struct.pack('i', len(input_shape)))
                if is_verbose:
                    print('Verbose - Number of dimensions of the input tensor: {}'.format(len(input_shape)))
                
                # write input dimension
                for i, dim in enumerate(input_shape):
                    file.write(struct.pack('i', dim))
                    if is_verbose:
                        print('Verbose - Dimension of index {0}: {1}'.format(i, dim))
                        
                # write input quantizer default: 0 (for float)
                file.write(struct.pack('i', 0))
                if is_verbose:
                    print('Verbose - Quantizer: 0')
            if name_node in dict_nodes_output:
                for name_node_out in dict_nodes_output[name_node]:
                    write_tree(name_node_out)
        write_tree(list_names_nodes_input[0])

def create_mapping_equivalent(graph):
    """Creates a dictionary mapping each node name to its equivalent name, i.e. ignoring the identities.
    
    Parameters
    ----------
    graph : tensorflow.core.framework.graph_pb2.GraphDef
        Graph definition.
    
    Returns
    -------
    dict
        Mapping between each node name and its equivalent
        name, i.e. ignoring the identities.
    
    """
    dict_nodes_equivalent = {}
    for node in graph.node:
        dict_nodes_equivalent[node.name] = remove_identity(graph,
                                                           node.name)
    return dict_nodes_equivalent

def create_mapping_input_outputs(graph, dict_nodes_equivalent):
    """Creates a dictionary mapping each input node name to its output nodes names, ignoring the identities.
    
    Parameters
    ----------
    graph : tensorflow.core.framework.graph_pb2.GraphDef
        Graph definition.
    dict_nodes_equivalent : dict
        Mapping between each node name and its equivalent
        name, ignoring the identities.
    
    Returns
    -------
    dict
        Mapping between each input node name and its
        output nodes names, ignoring the identities.
    
    """
    dict_nodes_output = {}
    for node in graph.node:
        for name_node_in in node.input:
            if node.op != 'Identity':
                
                # If the key `dict_nodes_equivalent[name_node_in]`
                # exists in `dict_nodes_output`, the key value is
                # returned. Otherwise, the key `dict_nodes_equivalent[name_node_in]`
                # with value [] are added to `dict_nodes_output`.
                dict_nodes_output.setdefault(dict_nodes_equivalent[name_node_in], []).append(node.name)
    return dict_nodes_output

def create_mapping_input_shape(list_names_nodes_input, list_nodes_input):
    """Creates a dictionary mapping each input node name to its shape.
    
    Parameters
    ----------
    list_names_nodes_input : list
        `list_names_nodes_input[i]` is the name of the input
        node of index i.
    list_nodes_input : list
        `list_nodes_input[i]` is the input node of index i.
    
    Returns
    -------
    dict
        Dictionary mapping each input node name to its shape.
    
    """
    dict_input_shape = {}
    for input_name in list_names_nodes_input:
        for input_tensor in list_nodes_input:
            if input_name in input_tensor.name:
                dict_input_shape[input_name] = input_tensor.shape
    return dict_input_shape

def create_mapping_name_id(graph, dict_nodes_equivalent, dict_nodes_output, name_node_input):
    """Creates a dictionary mapping each node name to a unique integer ID.
    
    Parameters
    ----------
    graph : tensorflow.core.framework.graph_pb2.GraphDef
        Graph definition.
    dict_nodes_equivalent : dict
        Mapping between the name of each node to
        itself if the node operation is not the
        identity.
    dict_nodes_output : dict
        Mapping between each input node name and
        its output nodes names, ignoring the identities.
    name_node_input : str
        Name of a node at the input to the graph.
    
    Returns
    -------
    dict
        Dictionary mapping each node name to a unique integer ID.
    
    Raises
    ------
    ValueError
        If the graph definition does not contain a node of the
        given name.
    
    """
    dict_nodes_ids = {}
    dict_counter = {'counter': 0}
    
    def fill_dictionary_ids(name_node):
        if name_node in dict_nodes_ids:
            return
        
        # The first argument of `next` is a generator.
        # This iterator yields the graph node whose name
        # matches `name_node`.
        node = next((node_search for node_search in graph.node if node_search.name == name_node),
                    None)
        if node is None:
            raise ValueError('The graph definition does not contain a node of name {}.'.format(name_node))
        
        # An ID is created for each node at the input to
        # the current node.
        for name_node_in in node.input:
            fill_dictionary_ids(dict_nodes_equivalent[name_node_in])
        if name_node in dict_nodes_ids:
            return
        dict_nodes_ids[name_node] = dict_counter['counter']
        dict_counter['counter'] += 1
        
        # An ID is created for each node at the output of
        # the current node.
        if name_node in dict_nodes_output:
            for name_node_out in dict_nodes_output[name_node]:
                fill_dictionary_ids(name_node_out)

    fill_dictionary_ids(name_node_input)
    return dict_nodes_ids

def remove_identity(graph, name_node_target):
    """Returns the name of the node at the input to the target node if the target node is an identity.
    
    Parameters
    ----------
    graph : tensorflow.core.framework.graph_pb2.GraphDef
        Graph definition.
    name_node_target : str
        Name of the target node.
    
    Returns
    -------
    str
        Name of the node at the input to the target node
        if the target node is an identity. If the target
        node is not an identity, its name is returned.
    
    Raises
    ------
    ValueError
        The graph definition does not contain a node of name `name_node_target`.
    ValueError
        The node of name `name_node_target` is an identity
        but it does not have one input.
    
    """
    # The first argument of `next` is a generator.
    # This iterator yields the graph node whose name
    # matches `name_node_target`.
    node = next((node_search for node_search in graph.node if node_search.name == name_node_target),
                None)
    if node is None:
        raise ValueError('The graph definition does not contain a node of name {}.'.format(name_node_target))
    if node.op == 'Identity':
        if len(node.input) != 1:
            raise ValueError('The node of name {} is an identity but it does not have one input.'.format(name_node_target))
        return remove_identity(graph,
                               node.input[0])
    else:
        return name_node_target

def write_attribute(node, file):
    """Writes the attribute of the graph node to the binary file.
    
    Parameters
    ----------
    node : tensorflow.core.framework.node_def_pb2.NodeDef
        Graph node.
    file: _io.BufferedWriter
        Writer of the binary file.
    
    Raises
    ------
    ValueError
        If, in 'BiasAdd', the data format is not NHWC.
    ValueError
        If, in 'MaxPool', the data format is not NHWC.
    ValueError
        If, in 'MaxPool', the padding is not SAME.
    ValueError
        If, in 'MatMul', an input tensor is transposed.
    ValueError
        If, in 'Conv2D', the data format is not NHWC.
    ValueError
        If, in 'Conv2D', the padding is not SAME.
    ValueError
        If, in 'ConcatV2', the number of concatenated tensors is
        not equal to 2.
    
    """
    if node.op == 'Const':
        write_characteristics_const(node.attr['value'].tensor,
                                    file)
    else:
        if node.op == 'BiasAdd':
            # `type(node.attr['data_format'].s)` is `bytes`.
            if node.attr['data_format'].s != b'NHWC':
                raise ValueError('In \'BiasAdd\', the data format is not NHWC.')
        elif node.op == 'MaxPool':
            if node.attr['data_format'].s != b'NHWC':
                raise ValueError('In \'MaxPool\', the data format is not NHWC.')
            if node.attr['padding'].s != 'SAME':
                raise ValueError('In \'MaxPool\', the padding is not SAME.')
            
            # The strides and the kernel sizes of the max-pooling
            # are written to the binary file.
            write_list(node.attr['strides'].list,
                       file)
            write_list(node.attr['ksize'].list,
                       file)
        elif node.op == 'MatMul':
            if node.attr['transpose_a'].b or node.attr['transpose_b'].b:
                raise ValueError('In \'MatMul\', an input tensor is transposed.')
        elif node.op == 'Conv2D':
            if node.attr['data_format'].s != b'NHWC':
                raise ValueError('In \'Conv2D\', the data format is not NHWC.')
            if node.attr['padding'].s != b'SAME':
                raise ValueError('In `\'Conv2D\', the padding is not SAME.')
            
            # The strides of the convolution are written to
            # the binary file.
            write_list(node.attr['strides'].list,
                       file)
        elif node.op == 'ConcatV2':
            if node.attr['N'].i != 2:
                raise ValueError('In \'ConcatV2\', the number of concatenated tensors is not equal to 2.')
        if node.op in ('Conv2D', 'MatMul', 'Mul'):
            file.write(struct.pack('i', 0)) # write internal quantizer

def write_characteristics_const(tensorproto, file):
    """Writes all the characteristics of the constant node.
    
    Parameters
    ----------
    tensorproto : tensorflow.core.framework.tensor_pb2.TensorProto
        Tensor protocol buffer object to be written to
        the binary file.
    file : _io.BufferedWriter
        Writer of the binary file.
    
    """
    nb_dims = len(tensorproto.tensor_shape.dim)
    if nb_dims:
        
        # The number of dimensions of the tensor are
        # written to the binary file.
        file.write(struct.pack('i', nb_dims))
        
        # The shape of the tensor is written to the
        # binary file.
        for i in range(nb_dims):
            file.write(struct.pack('i', tensorproto.tensor_shape.dim[i].size))
    else:
        file.write(struct.pack('i', 1))
        file.write(struct.pack('i', 1))
    
    # The content of the tensor is written to the binary file.
    write_tensorproto(tensorproto,
                      file)

def write_graph_global_infos(list_names_nodes_input, list_names_nodes_output, dict_nodes_ids, file, is_verbose):
    """Writes the global information of the graph to the binary file.
    
    Parameters
    ----------
    list_names_nodes_input : list
        Names of the nodes fed into the graph.
    list_names_nodes_output : list
        Names of the nodes at the graph output.
    dict_nodes_ids : dict
        Mapping between each node name to a unique integer ID.
    file : _io.BufferedWriter
        Writer of the binary file.
    is_verbose : bool
        If True, more verbose is set.
    
    """
    file.write(b'SADL0001')
    if is_verbose:
        print('Verbose - SADL0001')
    
    # output of the network type   0: int32 1: float 2: int16 default: float(1)
    file.write(struct.pack('i', int(1)))
    if is_verbose:
        print('Verbose - Type of the network: {}'.format(1))
    
    # The number of nodes in the graph is written
    # to the binary file.
    file.write(struct.pack('i', len(dict_nodes_ids)))
    if is_verbose:
        print('Verbose - Number of nodes: {}'.format(len(dict_nodes_ids)))
    
    # The number of inputs to the graph is written
    # to the binary file.
    file.write(struct.pack('i', len(list_names_nodes_input)))
    if is_verbose:
        print('Verbose - Number of input nodes: {}'.format(len(list_names_nodes_input)))
    
    # The IDs of the input nodes are written to
    # the binary file.
    for i in range(len(list_names_nodes_input)):
        file.write(struct.pack('i', dict_nodes_ids[list_names_nodes_input[i]]))
        if is_verbose:
            print('Verbose - ID of the input node of index {0}: {1}'.format(i, dict_nodes_ids[list_names_nodes_input[i]]))
    
    # The number of output nodes is written to
    # the binary file.
    file.write(struct.pack('i', len(list_names_nodes_output)))
    if is_verbose:
        print('Verbose - Number of output nodes: {}'.format(len(list_names_nodes_output)))
    
    # The IDs of the output nodes are written to
    # the binary file.
    for i in range(len(list_names_nodes_output)):
        file.write(struct.pack('i', dict_nodes_ids[list_names_nodes_output[i]]))
        if is_verbose:
            print('Verbose - ID of the output node of index {0}: {1}'.format(i, dict_nodes_ids[list_names_nodes_output[i]]))

def write_list(list_value_pb2, file):
    """Writes a Tensorflow attribute list value to the binary file.
    
    Parameters
    ----------
    list_value_pb2 : tensorflow.core.framework.attr_value_pb2.ListValue
        Tensorflow attribute list value.
    file : _io.BufferedWriter
        Writer of the binary file.
    
    """
    # The standard size of an integer is 4 bytes.
    file.write(struct.pack('i', len(list_value_pb2.i)))
    for i in range(len(list_value_pb2.i)):
        file.write(struct.pack('i', list_value_pb2.i[i]))

def write_tensorproto(tensorproto, file):
    """Writes the tensor protocol buffer object to the binary file.
    
    Parameters
    ----------
    tensorproto : tensorflow.core.framework.tensor_pb2.TensorProto
        Tensor protocol buffer object to be written to
        the binary file.
    file : _io.BufferedWriter
        Writer of the binary file.
    
    """
    # If `tensorproto.dtype` does not belong to {`tf.int32`, `tf.float32`},
    # a `KeyError` exception is raised.
    int_dtype_tf2cpp = csts.DICT_DTYPE_TENSORFLOW_DTYPE_TF2CPP[tensorproto.dtype]
    file.write(struct.pack('i', int_dtype_tf2cpp))
    if int_dtype_tf2cpp == 0:
        file.write(struct.pack('i', 0))
    
    # If `len(tensorproto.tensor_content)` is equal to 0,
    # the tensor protocol buffer object contains a single
    # coefficient.
    if len(tensorproto.tensor_content):
        file.write(tensorproto.tensor_content)
    else:
        if int_dtype_tf2cpp == 1:
            file.write(struct.pack('f', tensorproto.float_val[0]))
        else:
            file.write(struct.pack('i', tensorproto.int_val[0]))


