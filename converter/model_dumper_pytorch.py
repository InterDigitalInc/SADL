"""A library containing functions for parsing command-line arguments and options.
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
import torch
import torch.nn as nn
#from torchsummary import summary
#from torch.quantization import QuantStub, DeQuantStub
import numpy as np
import json
import struct
import re
import copy

import constants.constants_pytorch as csts


def parse_model_graph(model_graph, model, verbose):
    graph = dict()
    if verbose:
        print(model_graph)
    _model = {t[0]: t[1] for t in model.named_modules()}

    #print(model_graph.split("\n")[2:])

    graph_data = model_graph.split("\n")
    graph[graph_data[1].split(":", 1)[0].strip()] = ["input", []]
    for el in graph_data[2:-1]:
        arguments = re.search(r"(?s:.*)\((.*?)\)", el).group(1).split(", ")
        if "return" not in el:
            tmp = el.split("=", 1)
            id = tmp[0].split(":")[0].strip()
            if "GetAttr" in tmp[1]:
                name = "alias"
                t_alias = re.search(r"GetAttr\[name\=\"(.*?)\"\]", tmp[1]).group(1)
                component_name = ""
                if t_alias in _model.keys():
                    type = "none"
                    m = _model[t_alias]
                    if "Conv2d" in str(m):
                        type = "conv2d"
                        kernel_size = m.kernel_size
                        stride = m.stride
                        padding = m.padding
                        in_channels = m.in_channels
                        out_channels = m.out_channels
                        bias = m.bias.detach().numpy()
                        weight = m.weight.detach().numpy()
                        d = {"in": in_channels, "out": out_channels, "kernel": kernel_size, "stride": stride,
                             "padding": padding, "weight": weight, "bias": bias}
                    if "Linear" in str(m):
                        type = "linear"
                        in_features = m.in_features
                        out_features = m.out_features
                        bias = m.bias.detach().numpy()
                        weight = m.weight.detach().numpy()
                        d = {"in": in_features, "out": out_features, "weight": weight, "bias": bias}
                    if "ReLU" in str(m):
                        type = "relu"
                    if "MaxPool2d" in str(m):
                        type = "MaxPool"
                        kernel_size = m.kernel_size
                        stride = m.stride
                        padding = m.padding
                        d = {"kernel": kernel_size, "stride": stride, "padding": padding}
                    component = {"name": t_alias, "type": type, "data": d}
                arguments = [t_alias, component]
            if "CallMethod" in tmp[1]:
                name = re.search(r"CallMethod\[name\=\"(.*?)\"\]", tmp[1]).group(1)
            if "relu" in tmp[1]:
                name = "relu"
            if "add" in tmp[1]:
                name = "add"
                if len(arguments)>2:
                    if (graph[arguments[-1]][0] == 'constant'):
                        graph.pop(arguments[-1])
                        arguments.pop(-1)
            if "mul" in tmp[1]:
                name = "mul"
                if len(arguments)>2:
                    if (graph[arguments[-1]][0] == 'constant'):
                        graph.pop(arguments[-1])
                        arguments.pop(-1)
            if "flatten(" in tmp[1] or "view(" in tmp[1] or "reshape(" in tmp[1]:
                name = "reshape"
                # currently we support only flatten to 1d array BCHW => BK, K=C*H*W
                # constants can control dimensions to be flattened
                # for flatten it is difficult to get the dimensions
                shape = []
                layer_to_be_reshaped = ""
                const_to_be_used = list()
                # we have to hack it a lot in order to keep the same data in pytorch and cppCNN
                # in the pytorch model before reshape another reshape is called
                # to pack the data in the same way as cppCNN

                while len(arguments)>0:
                    if (graph[arguments[-1]][0] == 'list_construct'):
                        b_all_const = True
                        tmp_list = []
                        for c_ref in graph[arguments[-1]][1]:
                            if graph[c_ref][0] == 'constant':
                                b_all_const = b_all_const and True
                                tmp_list.append(int(graph[c_ref][1]))
                                graph.pop(c_ref)
                            else:
                                b_all_const = b_all_const and False
                                if graph[c_ref][0] == 'list_construct':
                                    for c1_ref in graph[c_ref][1]:
                                        graph.pop(c1_ref)
                                    graph.pop(c_ref)
                                else:
                                    layer_to_be_reshaped = c_ref

                            if b_all_const:
                                shape = tmp_list


                        graph.pop(arguments[-1])
                        arguments.pop(-1)
                arguments.append(layer_to_be_reshaped)
                arguments.append(shape)
            if "Constant" in tmp[1]:
                name = "constant"
                arguments = re.search(r"\[value\=(.*?)\]", tmp[1]).group(1)
            if "ListConstruct" in tmp[1]:
                name = "list_construct"
            if "cat(" in tmp[1]:
                name = "concatanate"
            if "ListUnpack" in tmp[1]:
                name = "list_unpack"
                r = [x.strip() for x in re.findall(r"\%(.*?)\:", tmp[0])]
                if graph[arguments[0]][0] == 'input':
                    graph.pop(arguments[0])
                    for e in r:
                        graph["%{}".format(e)] = [e, []]
                continue

        else:
            id = "%network-output"
            name = "return"

        graph[id] = [name, arguments]

    return graph

def convert_dim_pt_tf(tensor):
    shape_in = tensor.shape
    if len(shape_in) == 2:
        return shape_in
    elif len(shape_in) == 4:
        return tensor.permute(0, 2, 3, 1).shape

def get_seq_exec_list(model, input):
    model.eval()
    traced = torch.jit.trace(model, (input,), check_trace=False)
    return str(traced.graph)


def dumpModel(model, input_tensors, output_filename, weights, verbose):
    graph = parse_model_graph(get_seq_exec_list(model, input_tensors), model, verbose)
    # set ids
    id = 0
    id_names = dict()
    layer_name = {}
    layer_op = {}
    layer_padding = {}
    layer_stride = {}
    layer_op = {}
    layer_data = {}
    layer_data_quantizer = {}
    layer_inputs = {}
    layer_alias = dict()
    input_dimensions = dict()

    inputs_id = []
    outputs_id = []

    #for L in prms['config']['layers']:
    for k, L in graph.items():
        """
        layer_inputs[id] = []
        if len(L['inbound_nodes']) > 0:
            for i in L['inbound_nodes'][0]:
                layer_inputs[id].append(i[0])
        """
        if "input" in L[0]:
            name = L[0]
            layer_name[id] = name
            id_names[name] = id
            layer_alias[name] = name  # use when we replace some layers by another one (eg conv2D)
            layer_op[id] = 'Placeholder'
            layer_inputs[id] = []
            graph[k].append(id)
            inputs_id.append(id)
            input_dimensions[id] = convert_dim_pt_tf(input_tensors[len(inputs_id)-1])

        elif L[0] == "alias":
            name = L[1][0]
            layer_name[id] = name
            id_names[name] = id
            layer_alias[name] = name  # use when we replace some layers by another one (eg conv2D)
            layer_op[id] = L[1][1]["type"]
            layer_inputs[id] = []
            graph[k].append(id)
            #create layers with weights and biases
        elif L[0] == "forward":
            inputs = L[1]
            references = list()
            for it, i in enumerate(inputs):
                I = graph[i]
                if len(I) > 2: #check if have already proceed the alias
                    ref_id = I[2]
                    references.append(layer_name[ref_id])
                    if I[0] == "alias":
                        if I[1][1]["type"] in ['conv2d', 'linear']:
                            op_type = I[1][1]["type"]
                            #rewrite conv with conv_w, conv_c conv_b and conv_addb
                            # rewrite linear with linear_w, matmul, bias and addbiass
                            org_name = layer_name[ref_id]
                            name = org_name + "_w"
                            layer_name[ref_id] = name
                            layer_op[ref_id] = 'Const'
                            id_names[name] = ref_id
                            t_id = id
                            layer_alias[org_name] = name
                            w_tmp = I[1][1]["data"]["weight"]

                            if I[1][1]["type"] == "conv2d":
                                w_tmp = np.transpose(w_tmp, (2, 3, 1, 0))
                            else:
                                w_tmp = np.moveaxis(w_tmp, -2, -1)
                            layer_data[ref_id] = w_tmp

                            name = org_name + ("_c" if op_type == "conv2d" else "_m")
                            layer_name[id] = name
                            id_names[name] = id
                            layer_alias[name] = name  # use when we replace some layers by another one (eg conv2D)
                            layer_op[id] = 'Conv2D' if op_type == "conv2d" else "MatMul"
                            layer_alias[org_name] = org_name

                            if layer_op[id] == 'Conv2D':
                                layer_padding[id] = I[1][1]["data"]["padding"]
                                layer_stride[id] = I[1][1]["data"]["stride"]

                            # bias conv2D
                            if "bias" in I[1][1]["data"].keys() and len(I[1][1]["data"]["bias"]) > 0:
                                id = id + 1
                                name = org_name + '_b'
                                id_names[name] = id
                                layer_name[id] = name
                                layer_alias[name] = name
                                layer_inputs[id] = []
                                layer_op[id] = 'Const'
                                layer_data[id] = I[1][1]["data"]["bias"]  # layer_data[id]=model.get_layer('conv2d')

                                id = id + 1
                                name = org_name + '_badd'
                                id_names[name] = id  # biasadd has 2 inputs: output of conv2d and bias
                                layer_name[id] = name
                                layer_alias[name] = name
                                layer_inputs[id] = [org_name + "_b", org_name + ("_c" if op_type == "conv2d" else "_m")]
                                layer_op[id] = 'Add'

                        if I[1][1]["type"] == 'MaxPool':
                            if it == 0: # delete inputs only when we rewrite the forward for maxpool and not when maxpool is refernced
                                references = []
                                layer_inputs[id] = []
                            layer_name.pop(ref_id)
                            layer_op[id] = 'MaxPool'
                            t_id = id
                            layer_name[id] = I[1][0]
                            id_names[layer_name[id]] = id
                            layer_alias[layer_name[id]] = layer_name[id]  # use when we replace some layers by another one (eg conv2D)
                            graph[k].append(id)
                            strides = I[1][1]["data"]["stride"]
                            # strides length have to be 4
                            # strides for batch and channel have to be 1
                            # BHWC
                            if len(strides) == 2:
                                strides = (1, ) + strides + (1, )
                            elif len(strides) == 1:
                                strides = (1,) + strides + strides + (1,)
                            layer_stride[id] = strides

                            # As the maxpoolig layer has no padding,
                            # the kernel sizes are stored in the place
                            # dedicated to the padding.
                            kernel_size = I[1][1]["data"]["kernel"]
                            if len(kernel_size) == 2:
                                kernel_size = (1, ) + kernel_size + (1, )
                            elif len(kernel_size) == 1:
                                strides = (1,) + kernel_size + kernel_size + (1,)
                            layer_padding[id] = kernel_size
                        graph[k].append(id)

            layer_inputs[t_id] = references


        elif L[0] == "relu":
            input = L[1][0]
            I = graph[input]
            ref_id = I[2]
            ref_name = layer_name[ref_id]
            name = "relu_{}".format(ref_id)
            layer_op[id] = 'Relu'
            layer_inputs[id] = []
            layer_inputs[id].append(ref_name)
            id_names[name] = id
            layer_alias[name] = name
            layer_name[id] = name

            graph[k].append(id)

        elif L[0] == "list_construct":
            inputs = L[1]
            references = list()

            name = "list_construct"
            layer_name[id] = name
            id_names[name] = id
            layer_alias[name] = name  # use when we replace some layers by another one (eg conv2D)
            layer_op[id] = 'Placeholder'
            layer_inputs[id] = []

            for i in inputs:
                I = graph[i]
                if len(I) == 3: #check if have already proceed the alias
                    ref_id = I[2]
                    references.append(layer_name[ref_id])
            layer_inputs[id] = references

            graph[k].append(id)

        elif L[0] == "constant":
            layer_data[id] = np.array([int(L[1])], dtype='int32')
            layer_data_quantizer[id] = 1
            layer_inputs[id] = []
            layer_op[id] = 'Const'
            graph[k].append(id)
            layer_name[id] = "constant_{}".format(id)
            layer_alias[layer_name[id]] = layer_name[id]
            id_names[layer_name[id]] = id

        elif L[0] == "concatanate":
            inputs = L[1]
            org_name = "concatanate"
            ref_id = -1
            cnctn_inputs = list()
            cnctn_data_id = -1
            id -= 1
            for i in inputs:
                I = graph[i]
                if len(I) == 3: #check if have already proceed the alias
                    if I[0] == "list_construct":
                        cnctn_inputs = layer_inputs[I[2]]
                        cnctn_data_id = I[2]

                    if I[0] == "constant":
                        cnctn_axis_id = I[2]
                        id_names[org_name + '_axis'] = cnctn_data_id
                        layer_name[cnctn_data_id] = org_name + '_axis'
                        layer_alias[layer_name[cnctn_data_id]] = layer_name[cnctn_data_id]
                        layer_op[cnctn_data_id] = 'Const'
                        layer_inputs[cnctn_data_id] = []
                        layer_data_quantizer[cnctn_data_id] = 1
                        layer_data[cnctn_data_id] = np.array([csts.DICT_MAPPING_PYTORCH_TF[int(I[1])]], dtype='int32')

                        layer_name[cnctn_axis_id] = org_name
                        layer_op[cnctn_axis_id] = 'ConcatV2'
                        id_names[org_name] = cnctn_axis_id
                        layer_alias[org_name] = org_name
                        graph[k].append(cnctn_axis_id)
                        layer_inputs[cnctn_axis_id] = [org_name + '_axis'] + cnctn_inputs[::-1]




        elif L[0] == "add":
            layer_op[id] = 'Add'
            inputs = L[1]
            references = list()
            org_name = "add_{}".format(id)
            for i in inputs:
                I = graph[i]
                if len(I) > 2: #check if have already proceed the alias
                    ref_id = I[2]
                    references.append(layer_name[ref_id])
            layer_inputs[id] = references
            id_names[org_name] = id
            layer_name[id] = org_name
            layer_alias[layer_name[id]] = layer_name[id]
            graph[k].append(id)

        elif L[0] == "mul":
            layer_op[id] = 'Mul'
            inputs = L[1]
            references = list()
            org_name = "mul_{}".format(id)
            for i in inputs:
                I = graph[i]
                if len(I) == 3: #check if have already proceed the alias
                    ref_id = I[2]
                    references.append(layer_name[ref_id])
            layer_inputs[id] = references
            id_names[org_name] = id
            layer_name[id] = org_name
            layer_alias[layer_name[id]] = layer_name[id]
            graph[k].append(id)

        elif L[0] == "reshape":
            org_name = L[0]
            id_names[org_name + '_shape'] = id
            layer_name[id] = org_name + '_shape'
            layer_alias[layer_name[id]] = layer_name[id]

            layer_inputs[id + 1] = [org_name + '_shape']
            layer_inputs[id + 1].append(layer_name[graph[L[1][0]][2]])
            layer_inputs[id] = []
            layer_op[id] = 'Const'
            shape = L[1][1]
            layer_data[id] = np.asarray(shape, dtype='int32')
            layer_data_quantizer[id] = 1

            id = id + 1
            layer_op[id] = 'Reshape'
            id_names[org_name] = id
            layer_name[id] = org_name
            layer_alias[layer_name[id]] = layer_name[id]
            graph[k].append(id)


        elif L[0] == "return":
            for el in L[1]:
                outputs_id.append(graph[el][2])

        id = id + 1


    # dump
    file_out = False
    if output_filename is not None:
        file_out = True
        fd = open(output_filename, 'wb')
        fd.write(b'SADL0001')
        # output of the network type 0: int32 1: float 2: int16 default: float(1)
        fd.write(struct.pack('i', int(1)))

    print("# Nb layers: ", len(layer_name))
    if file_out: fd.write(struct.pack('i', int(len(layer_name))))

    print("# Nb inputs: ", len(inputs_id))
    if file_out: fd.write(struct.pack('i', int(len(inputs_id))))

    for i in inputs_id:
        print("#  input ", i)
        if file_out: fd.write(struct.pack('i', int(i)))

    print("# Nb outputs: ", len(outputs_id))
    if file_out: fd.write(struct.pack('i', int(len(outputs_id))))

    for i in outputs_id:
        print("#  output ", i)
        if file_out: fd.write(struct.pack('i', int(i)))

    print("")


    for id in layer_name.keys():
        print("# Layer id ", id)
        if file_out: fd.write(struct.pack('i', int(id)))

        print("#  op ", layer_op[id])
        if file_out: fd.write(struct.pack('i', int(csts.DICT_FIELD_NUMBERS[layer_op[id]])))

        print("#  name_size ", len(layer_name[id]))
        if file_out: fd.write(struct.pack('i', int(len(layer_name[id]))))

        print("#  name ", layer_name[id])
        if file_out: fd.write(str.encode(layer_name[id]))

        print("#  nb_inputs ", len(layer_inputs[id]))
        if file_out: fd.write(struct.pack('i', int(len(layer_inputs[id]))))

        for i in layer_inputs[id][::-1]:
            print("#    ", id_names[layer_alias[i]], "(" + layer_alias[i] + ")")
            if file_out: fd.write(struct.pack('i', int(id_names[layer_alias[i]])))

        # custom data
        if layer_op[id] == 'Const':
            print("#  nb_dim", len(layer_data[id].shape))  # dim [int32_t]
            if file_out: fd.write(struct.pack('i', int(len(layer_data[id].shape))))

            for i in layer_data[id].shape:
                print("#   ", i)
                if file_out: fd.write(struct.pack('i', int(i)))

            data_type = -1
            if layer_data[id].dtype == 'float32':
                data_type = 1
            elif layer_data[id].dtype == 'int32':
                data_type = 0
            elif layer_data[id].dtype == 'int16':
                data_type = 2
            elif layer_data[id].dtype == 'int8':
                data_type = 3
            else:
                print("unkwown type")
                quit()
            print("#  dtype", data_type)  # dim [int32_t]
            if file_out:  fd.write(struct.pack('i', int(data_type)))
            if data_type != 1:  # not float
                print("#  quantizer", int(layer_data_quantizer[id]))
                if file_out:  fd.write(struct.pack('i', int(layer_data_quantizer[id])))
            if file_out: fd.write(layer_data[id].tobytes())

        if layer_op[id] == 'Conv2D':
            print("#  nb_dim_strides", len(layer_stride[id]))  # dim [int32_t]
            if file_out: fd.write(struct.pack('i', int(len(layer_stride[id]))))

            for i in layer_stride[id]:
                print("#   ", i)
                if file_out: fd.write(struct.pack('i', int(i)))

        if layer_op[id] == 'MaxPool':
            print("#  nb_dim_strides", len(layer_stride[id]))  # dim [int32_t]
            if file_out: fd.write(struct.pack('i', int(len(layer_stride[id]))))

            for i in layer_stride[id]:
                print("#   ", i)
                if file_out: fd.write(struct.pack('i', int(i)))
            print("# nb_dim_kernels", len(layer_padding[id]))
            if file_out: fd.write(struct.pack('i', int(len(layer_padding[id]))))

            for i in layer_padding[id]:
                print("#   ", i)
                if file_out: fd.write(struct.pack('i', int(i)))

        if layer_op[id] in ("Conv2D", "Mul", "MatMul"):
            # output the internal quantizer default: 0
            if file_out: fd.write(struct.pack('i', int(0)))

        if layer_op[id] == 'Placeholder':
            print("#  nb input dimensions", len(input_dimensions[id]))
            if file_out: fd.write(struct.pack('i', int(len(input_dimensions[id]))))
            for i in input_dimensions[id]:
                print("#    ", i)
                if file_out: fd.write(struct.pack('i', int(i)))
            # output the quantizer of the input default: 0
            print("#   quantizer_of_input", 0)
            if file_out: fd.write(struct.pack('i', int(0)))

        print("")
    print("TODO: check data order")

    if file_out:
        fd.close()
        print("[INFO] dumped model in " + output_filename)

