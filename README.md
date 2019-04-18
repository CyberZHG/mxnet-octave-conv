# MXNet Octave Conv

[![Travis](https://travis-ci.org/CyberZHG/mxnet-octave-conv.svg)](https://travis-ci.org/CyberZHG/mxnet-octave-conv)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/mxnet-octave-conv/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/mxnet-octave-conv)
[![996.ICU](https://img.shields.io/badge/license-Anti%20996-blue.svg)](https://996.icu) 

Unofficial implementation of [Drop an Octave: Reducing Spatial Redundancy in
Convolutional Neural Networks with Octave Convolution](https://arxiv.org/pdf/1904.05049.pdf).

## Install

```bash
pip install mxnet-octave-conv
```

## Usage

```python
import mxnet as mx
from mxnet_octave_conv import octave_conv, octave_dual

mx.symbol.Variable(name='data')
conv = octave_conv(x, num_filter=7, kernel=(3, 3))
pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
conv = octave_conv(pool, num_filter=5, kernel=3, stride=1, dilate=(2, 3), name='Mid')
pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
conv = octave_conv(pool, num_filter=3, kernel=3, stride=(1, 1), dilate=1, ratio_out=0.0)
pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
flatten = mx.symbol.Flatten(pool)
dense = mx.symbol.FullyConnected(flatten, num_hidden=2)
model = mx.symbol.SoftmaxOutput(dense, name='softmax')
print(mx.visualization.print_summary(model, shape={'data': (2, 3, 32, 32)}))
```
