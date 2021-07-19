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

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import importlib

def quantizer_inputs_10_bits(inputs, quantizer):
    scale = np.math.pow(2, quantizer)
    list_inputs_quantized = []
    for i in range(len(inputs)):
        k = (inputs[i] * scale).to(torch.int)
        list_inputs_quantized.append(k.to(torch.float) / scale)
    return list_inputs_quantized
    
def load_model(net, modelPath):
    # optionally copy weights from a checkpoint
    weights = torch.load(modelPath, map_location=lambda storage, loc: storage)
    # net.load_state_dict(weights['model'].state_dict())
    net.load_state_dict(weights['state_dict'])
    return net

os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../tests_data')
sys.path.append(os.getcwd())

import model_dumper_pytorch as md

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
    parser.add_argument('--verbose',
                        action='store_true')
    args = parser.parse_args()

    if args.input_py:
        test = importlib.import_module(args.input_py, package=None)
        model = test.getModel()
        inputs = test.getInput()
    else:
       raise('[ERROR] You should specify a py file') 
       quit()

    if args.verbose: print(model)

    weights = {}
    md.dumpModel(model, inputs,  args.output, weights,  args.verbose)

    # The list of inputs Numpy arrays is overwritten using
    # the list of their quantized version over 10 bits.
    if args.output_results is not None:
        inputs = quantizer_inputs_10_bits(inputs, 10)
        output = model(inputs)
        with open(args.output_results, 'w') as f:
            f.write(str(len(inputs)) + '\n')
            for input in inputs:
                input = input.detach().numpy()
                if len(input.shape) == 4:
                    input = np.transpose(input, (0, 2, 3, 1))

                f.write(str(len(input.shape)) + '\n')
                for i in input.shape:
                    f.write("{} ".format(i))
                f.write('\n')
                for x in np.nditer(input, order='C'):
                    f.write("{} ".format(x))
                f.write('\n')
            f.write('1\n')
            output = output.detach().numpy()
            if len(output.shape) == 4:
                output = np.transpose(output, (0, 2, 3, 1))
            f.write(str(len(output.shape)) + '\n')
            for i in output.shape:
                f.write("{} ".format(i))
            f.write('\n')
            for x in np.nditer(output, order='C'):
                f.write("{} ".format(x))
            f.write('\n')
        print("[INFO] results file in", args.output_results)


