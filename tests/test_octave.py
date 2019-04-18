import logging
from unittest import TestCase
import mxnet as mx
import numpy as np
from mxnet_octave_conv import octave_conv, octave_dual, octave_residual


class TestMXNetOctave(TestCase):

    def _test_fit(self, symbol):
        logging.getLogger().setLevel(logging.DEBUG)
        data_size = 4096
        batch_size = 32
        x = np.random.standard_normal((data_size, 3, 32, 32))
        y = np.random.randint(0, 2, data_size)
        data = mx.io.NDArrayIter(x, y, batch_size=batch_size)
        model = mx.model.FeedForward(
            symbol=symbol,
            num_epoch=5,
            optimizer='adam',
            numpy_batch_size=batch_size,
        )
        model.fit(
            X=data,
            batch_end_callback=mx.callback.Speedometer(batch_size, 200),
        )
        predict = model.predict(x).argmax(axis=-1)
        self.assertLess(np.sum(np.abs(y - predict)), data_size // 2)

    def test_fit_basic(self):
        x = mx.symbol.Variable(name='data')
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
        self._test_fit(model)

    def test_fit_low(self):
        x = mx.symbol.Variable(name='data')
        conv = octave_conv(x, num_filter=7, kernel=(3, 3))
        pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
        conv = octave_conv(pool, num_filter=5, kernel=3, stride=1, dilate=(2, 3), name='Mid')
        pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
        conv = octave_conv(pool, num_filter=3, kernel=3, stride=(1, 1), dilate=1, ratio_out=1.0)
        pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
        flatten = mx.symbol.Flatten(pool)
        dense = mx.symbol.FullyConnected(flatten, num_hidden=2)
        model = mx.symbol.SoftmaxOutput(dense, name='softmax')
        print(mx.visualization.print_summary(model, shape={'data': (2, 3, 32, 32)}))
        self._test_fit(model)

    def test_fit_residual(self):
        x = mx.symbol.Variable(name='data')
        conv = octave_conv(x, num_filter=8, kernel=(3, 3))
        conv_dual = octave_dual(conv, lambda data: mx.symbol.Convolution(data, num_filter=4, kernel=(3, 3), pad=(1, 1)))
        residual = octave_residual(conv, conv_dual)
        pool = octave_dual(
            residual,
            lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'),
        )
        conv = octave_conv(pool, num_filter=5, kernel=3, stride=1, dilate=(2, 3), name='Mid')
        pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
        conv = octave_conv(pool, num_filter=3, kernel=3, stride=(1, 1), dilate=1, ratio_out=1.0)
        pool = octave_dual(conv, lambda data: mx.symbol.Pooling(data, kernel=(2, 2), stride=(2, 2), pool_type='max'))
        flatten = mx.symbol.Flatten(pool)
        dense = mx.symbol.FullyConnected(flatten, num_hidden=2)
        model = mx.symbol.SoftmaxOutput(dense, name='softmax')
        print(mx.visualization.print_summary(model, shape={'data': (2, 3, 32, 32)}))
        self._test_fit(model)
