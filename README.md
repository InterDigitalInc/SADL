# SADL: Small Adhoc Deep-Learning Library

A small library to perform inference in pure C++.
Models from TF 1.x, TF 2.x and pytorch can be converted to a simple format compatible with the library.
Inference can be done completely in C++ without any external dependencies.


## Conversion instruction
Two converters are available:
- converter/main.py for TensorFlow (1.x or 2.x)
- converter/main_pytorch.py for pytorch

Example of conversion:
```shell
cd ../sample
python ../converter/main.py --input_py examples.tf2 --output tf2.model --output_results tf2.results
```
Where examples/tf2.py contains a model and an example of input.

The converter ouputs 2 files:
- tf2.model: a binary file containing the converted model
- tf2.results: the results of inference in python using the provided model in ascii format


## Build instruction
The library is header only and does not require to be build.
However, when integrated into a software, build options will drive several aspects:
- several macros will control the debug level and information level. The macros can be find in sadl/optiosn.h
- the simd options pass to the compiler will control the level of SIMD activated in the code (the level is not, in the library, control dynamically)

An example is given in the sample directory CMakeLists.txt:
- release version just contains the "fast" version without any check or instrumentation for complexity assessment
- debug version contains a more debug information to check the model validity, which layers are not SIMD optimized, the number of operations, the number of overflow in case of interger NN

Example of build:
```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ../sample
make
```


## Validation instruction
After converting the model and building the sample program, it can be used to validate the model inference in C++:
```shell
cd sample
../build/sample/sample_generic tf2.model tf2.results
```

### Output reading: model loading
This part shows the model load issues and information.
```shell
[INFO] Model loading information
[INFO] start model loading
[INFO] read magic SADL0001
[INFO] Model type: 1
[INFO] Num layers: 11
[INFO] input id: 0 
[INFO] output id: 10 
[INFO] id: 0 op  Placeholder
  - name: input0
  - inputs: 
  - dim: [ 1 3 8 8 ]
  - q: 0
...
[INFO] end model loading
```

### Output reading: model initialization
This part shows information or issues during initialization of the model.
```shell
[INFO] Model initilization information
[INFO] start model init
[INFO] float mode
[INFO] use swapped tensor
[INFO] inserted 0 copy layers
[INFO] reshape 1 conv2d_w [ 3 3 8 16 ] => [ 3 3 16 8 ]
[INFO] reshape 6 conv2d_1_w [ 3 3 16 32 ] => [ 3 3 32 16 ]
[INFO] init layer 0 Placeholder input0
[INFO] init layer 1 Const conv2d_w
[INFO] init layer 2 Conv2D conv2d_c
  - input conv2d: [ 1 3 8 8 ] [ 3 3 16 8 ]
  - output Conv2D: [ 1 3 8 16 ]
[INFO] init layer 3 Const conv2d_b
[INFO] init layer 4 Add conv2d_badd
  - [ 1 3 8 16 ] [ 16 ]
[INFO] init layer 5 Relu conv2d_a
[INFO] init layer 6 Const conv2d_1_w
[INFO] init layer 7 Conv2D conv2d_1_c
  - input conv2d: [ 1 3 8 16 ] [ 3 3 32 16 ]
  - output Conv2D: [ 1 3 8 32 ]
[INFO] init layer 8 Const conv2d_1_b
[INFO] init layer 9 Add conv2d_1_badd
  - [ 1 3 8 32 ] [ 32 ]
[INFO] init layer 10 Relu conv2d_1_a
[INFO] end model init
```

### Output reading: model inference
This part shows information or issues during inferene of the model.
```shell
[INFO] layer 0: ok
[INFO] layer 1 (Const): ok
[INFO] layer 2 (Conv2D): ok
[INFO] layer 3 (Const): ok
[INFO] layer 4 (Add): ok
[INFO] layer 5 (Relu): ok
[INFO] layer 6 (Const): ok
[INFO] layer 7 (Conv2D): ok
[INFO] layer 8 (Const): ok
[INFO] layer 9 (Add): ok
[INFO] layer 10 (Relu): ok
```

### Output reading: model inference
This part shows information on the number of operations in each layer (usually MAC but depends on the layer).
```shell
[INFO] Complexity assessment
[INFO] layer 2 conv2d_c [Conv2D]: 384 op
[INFO] layer 4 conv2d_badd [Add]: 384 op
[INFO] layer 7 conv2d_1_c [Conv2D]: 768 op
[INFO] layer 9 conv2d_1_badd [Add]: 768 op
[INFO] Total number of operations: 2304
[INFO] ---------------------------------
```

### Output reading: inference error
This part shows information on the output error compared to the python side inference.
```shell
[INFO] Error assessment
[INFO] test OK  Qout=0 max: max_error=1.78814e-07 (th=0.001), output: max |x|=0.614462 av|x|=0.0997989
```
In this example, the maximum error was 1.78814e-07 on all samples of the output. Scale of the output is also given as maximum and average.


## License
SADL is licensed under the BSD-3-Clause.


## Citation
If you use this software for publication, here is the BibTex entry:
```
@techreport{SADL2021,
      title       = "{SADL} Small Adhoc Deep-Learning Library",
      author      = "Franck Galpin and Pavel Nikitin and Thierry Dumas and Philippe Bordes",
      institution = "InterDigital",
      number      = "JVET-W0181",
      year        = 2021,
      month       = jul,
      url       = {https://jvet-experts.org/doc_end_user/current_document.php?id=11012}
}
```

