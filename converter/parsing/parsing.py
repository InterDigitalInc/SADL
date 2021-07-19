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

import argparse

def int_strictly_positive(string):
    """Converts the string into an integer.
    
    Parameters
    ----------
    string : str
        String to be converted into an integer.
    
    Returns
    -------
    int
        Integer resulting from the conversion.
    
    Raises
    ------
    ArgumentTypeError
        If the string cannot be converted into an integer.
    ArgumentTypeError
        If the integer resulting from the conversion is not
        strictly positive.
    
    """
    try:
        integer = int(string)
    except ValueError:
        raise argparse.ArgumentTypeError('\"{}\" cannot be converted into an integer.'.format(string))
    if integer <= 0:
        raise argparse.ArgumentTypeError('\"{}\" is not strictly positive.'.format(integer))
    else:
        return integer

def tuple_strings(string):
    """Converts the string into a tuple of strings.
    
    Parameters
    ----------
    string : str
        If `string` contains several non-empty
        strings separated by a comma, the output
        tuple contains these strings.
    
    Returns
    -------
    tuple
        Non-empty strings.
    
    """
    list_strings = string.split(',')
    list_cleaned = []
    for item in list_strings:
        if item:
            list_cleaned.append(item)
    return tuple(list_cleaned)


