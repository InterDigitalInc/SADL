#!/bin/bash


echo "[INFO] BUILD SADL SAMPLE"
# build sample
mkdir -p sample_test;
cd sample_test;
cmake -DCMAKE_BUILD_TYPE=Release ../sample
make
echo ""

echo "[INFO] TF2 -> ONNX -> SADL"
# TF2
python3 ../sample/tf2.py 2>/dev/null # output a tf2.onnx
python3 ../converter/main.py --input_onnx tf2.onnx --output tf2.sadl 
./sample_simd tf2.sadl | tail -2
echo ""

echo "[INFO] PYTORCH -> ONNX -> SADL"
# torch
python3 ../sample/pytorch.py 2>/dev/null # output a pytorch.onnx
python3 ../converter/main.py --input_onnx pytorch.onnx --output pytorch.sadl 
./sample_simd pytorch.sadl | tail -2

if [ -f tf2.sadl -a -f pytorch.sadl -a -f sample_generic -a -f sample_simd ]; then
 exit 0;
else
 exit 1;
fi;
