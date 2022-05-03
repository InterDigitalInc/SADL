# SADL: Small Adhoc Deep-Learning Library

A small library to perform inference in pure C++.
Models in ONNX format can be converted to a simple format compatible with the library.
ONNX export feature is supported by all majors framework (TF1.x, TF2.x, PyTorch etc.).
Inference can be done completely in C++ without any external dependencies.


## Conversion instruction
Conversion is performed from an ONNX file.
In the sample directory, 2 examples are given.
```python
import numpy as np
import os
import tf2onnx
import onnx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  
tensor_fmt = 'channels_last'
import tensorflow as tf

s = (16, 16,3)
inputs = tf.keras.Input(shape=s, name='input0', dtype=tf.float32)

nbf = 8 
x = tf.keras.layers.Conv2D(nbf, (3,3) , activation='linear',data_format=tensor_fmt, use_bias=True,bias_initializer="glorot_uniform",padding='same')(inputs)
x = tf.keras.layers.MaxPool2D(2,data_format=tensor_fmt)(x)
x = tf.keras.layers.Conv2D(nbf, (3,3) , activation='linear',data_format=tensor_fmt, use_bias=True,bias_initializer="glorot_uniform",padding='same')(x)

x0 = tf.keras.layers.Conv2D(nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x)
x0 = tf.keras.layers.Conv2D(nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x0)
x0 = x0 + x
x0 = tf.keras.layers.MaxPool2D(2,data_format=tensor_fmt)(x0)
x0 = tf.keras.layers.Conv2D(2*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x0)

x1 = tf.keras.layers.Conv2D(2*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x0)
x1 = tf.keras.layers.Conv2D(2*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x1)
x1 = x1 + x0
x1 = tf.keras.layers.MaxPool2D(2,data_format=tensor_fmt)(x1)
x1 = tf.keras.layers.Conv2D(4*nbf, kernel_size=(3, 3) , activation='relu', use_bias=True,data_format=tensor_fmt, padding='same')(x1)

x2 = tf.keras.layers.Reshape((1,4*nbf*16//8*16//8))(x1)
y = tf.keras.layers.Dense(2)(x2)
model = tf.keras.Model(inputs=[inputs],outputs=y,name="cat_classifier")


X = np.linspace(-1.,1,np.prod(s)).reshape((1,)+s)
Y = model(X)

model_onnx , _ = tf2onnx.convert.from_keras(model,[tf.TensorSpec(shape=(1,)+s,name="input0")],opset=13)
onnx.save(model_onnx, "./tf2.onnx")
print("Output\n",Y)
print("Model in tf2.onnx")
```

Example of conversion
```shell
# output a tf2.onnx
python3 sample/tf2.py 

# convert the onnx to sadl
python3 converter/main.py --input_onnx tf2.onnx --output tf2.sadl 
```

In PyTorch the same example is given:
```shell
# output a pytorch.onnx
python3 sample/pytorch.py 

# convert the onnx to sadl
python3 converter/main.py --input_onnx pytorch.onnx --output pytorch.sadl 
```
Please note taht frameworks using the NCHW data layout are changed to a NHWC data layout graph (one just need to adapt the inputs and/or inputs).

## Build instruction
The library is header only and does not require to be build.
However, when integrated into a software, build options will drive several aspects:
- several macros will control the debug level and information level. The macros can be find in sadl/optiosn.h
- the simd options pass to the compiler will control the level of SIMD activated in the code (the level is not, in the library, control dynamically)

Several examples are given in the sample directory CMakeLists.txt:
- count_mac: just assess the model complexity
- debug_model: print information on potential issues in the model (wrong values, no SIMD layers etc.)
- sample: inference of the model with 3 levels of optimization (generic, simd256, simd512)

Example of program:
```c++
#include <sadl/model.h>

int main() {
  sadl::Model<float> model;
  
  ifstream file("model.sadl", ios::binary);
  if (!model.load(file)) {
    cerr << "[ERROR] Unable to read model " << endl;
    exit(-1);
  }

  vector<sadl::Tensor<float>> inputs=model.getInputsTemplate();  
  if (!model.init(inputs)) {
    cerr << "[ERROR] issue during initialization" << endl;
    exit(-1);
  }
  
  if (!model.apply(inputs)) {
    cerr << "[ERROR] issue during inference" << endl;
    exit(-1);
  }
  const int N=model.nbOutput();
  for(int i=0;i<N;++i) cout<<"[INFO] output "<<i<<'\n'<<model.result(i)<<endl;
}
```

A script test is available to test all programs:
```shell
./sample/sample_test.sh
```


## Validation instruction


### Output reading: model loading
```shell
sample_test/debug_model tf2.sadl
```
This part shows the model load issues and information on all layers parameters.
```shell
[INFO] == start model loading ==
[INFO] start model loading
[INFO] read magic SADL0001
[INFO] Model type: 1
[INFO] Num layers: 38
[INFO] input id: 0 
[INFO] output id: 37 
[INFO] id: 0 op  Placeholder
  - name: input0
  - inputs: 
  - dim: ( 1 16 16 3 )
  - q: 0
[INFO] id: 1 op  Const
  - name: cat_classifier/conv2d/Conv2D/ReadVariableOp:0
  - inputs: 
  - tensor: ( 3 3 3 8 )
  - data: -0.230531 0.0269793 0.181556 -0.0948688  ...
  - quantizer: 0
[INFO] id: 2 op  Conv2D
  - name: cat_classifier/conv2d/BiasAdd:0
  - inputs: 0 1 
  - strides: ( 1 1 1 1 )
  - pads: ( 1 1 1 1 )
  - q: 0
[INFO] id: 3 op  Const
...
[INFO] == end model loading ==
```

### Output reading: model initialization
```shell
sample_test/debug_model tf2.sadl
```
This part shows information or issues during initialization of the model. It also shows the size of each inputs and outputs for all layers.
```shell
[INFO] == start model init ==
[INFO] start model init
[INFO] float mode
[INFO] use swapped tensor
[INFO] inserted 0 copy layers
...
[INFO] init layer 0 Placeholder input0
[INFO] init layer 1 Const cat_classifier/conv2d/Conv2D/ReadVariableOp:0
[INFO] init layer 2 Conv2D cat_classifier/conv2d/BiasAdd:0
  - input conv2d: ( 1 16 16 3 ) ( 3 3 8 3 )
  - output Conv2D: ( 1 16 16 8 )
...
[INFO] == end model init ==
```

### Output reading: model inference debugging
```shell
sample_test/debug_model tf2.sadl
```
This part shows information or issues during inference of the model. 
```shell
[INFO] == start model inference ==
[INFO] 0 input.1 [PlaceHolder]: q=0 [0 0 0 0 0 0 0 0 ]
[INFO] layer 0: ok
[INFO] 1 conv01.weight Const]: inputs=[... ] q=0 [0.191466 -0.0538487 0.132789 0.103821 0.0535023 -0.214573 -0.197487 -0.113809  ...]
[INFO] layer 1 (Const): ok
[INFO] 2 19 Conv2D]: inputs=[0 (q=0) 1 (q=0) ... ] 
[WARN] generic version conv 3x3 inD=3 outD=8 s=[1 1]  16x16 55 kMAC
q=0 [0 0 0 0 0 0 0 0  ...]
[INFO] layer 2 (Conv2D): ok
[INFO] 3 conv01.bias Const]: inputs=[... ] q=0 [0 0 0 0 0 0 0 0  ...]
...
[INFO] Inference OK
[INFO] == end model inference ==
```
To get a more readable list of issues:
```shell
sample_test/debug_model tf2.sadl | grep WARN
```
```shell
[WARN] generic version conv 3x3 inD=3 outD=8 s=[1 1]  16x16 55 kMAC
```
Here we see that the convolution using 3x3 kernel, a stride of 1 with a number of kernels of 8 is not SIMD accelerated.

### Output reading: model complexity
```shell
sample_test/debug_model tf2.sadl | grep WARN
```
This part shows information on the number of operations in each layer (usually MAC but depends on the layer and the SIMD level).
```shell
[INFO] Complexity assessment
[INFO] layer 2 cat_classifier/conv2d/BiasAdd:0 [Conv2D]: 2048 op
[INFO] layer 4 cat_classifier/conv2d/BiasAdd:0_NOT_IN_GRAPH [BiasAdd]: 2048 op
[INFO] layer 7 cat_classifier/conv2d_1/BiasAdd:0 [Conv2D]: 512 op
...
```

The next part shows the real number of MAC executed by the model:
```shell
[INFO] 0 overflow
[INFO] 7810 OPs
[INFO] 216160 MACs
[INFO] 216160 MACs non 0
```

### Output reading: inference error
```shell
sample_test/debug_model tf2.sadl 
```

This shows the output of the network. It can be compared to the python side inference.
```shell
...
[INFO] output 0
[ [[-0.178823	0.101747	  ] ]]
```

### Performance assessment 
The generic version inference is given by:
```shell
sample_test/sample_generic tf2.sadl 
...
[INFO] 6.80042 ms
...
```

The AVX512 version inference is given by:
```shell
sample_test/sample_simd512 tf2.sadl 
...
[INFO] 0.449472 ms
...
```


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

