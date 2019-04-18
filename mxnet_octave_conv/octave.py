import mxnet as mx

__all__ = ['octave_conv', 'octave_dual', 'octave_residual']


def _make_tuple(x, rank):
    if isinstance(x, int):
        return (x,) * rank
    return tuple(x)


def _make_name(name, suffix=''):
    if name is None:
        return None
    return name + '-' + suffix


def octave_conv(data,
                num_filter,
                kernel,
                octave=2,
                ratio_out=0.5,
                stride=1,
                dilate=1,
                num_group=1,
                workspace=1024,
                no_bias=0,
                cudnn_tune=None,
                cudnn_off=0,
                layout=None,
                name=None,
                attr=None,
                **kwargs):
    """Create octave convolution structure.

    Parameters
    ----------
    octave : int (positive), optional, default=2
        The division of the spatial dimensions by a power of 2.
    ratio_out: float (non-negative), optional, default=0.5
        The ratio of filters for lower spatial resolution.

    Returns
    -------
    Symbol(tuple)
        The result symbols. Only one symbol will be returned if ratio_out is either 0.0 or 1.0.

    References
    ----------
    - [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution]
      (https://arxiv.org/pdf/1904.05049.pdf)
    """
    if isinstance(data, (list, tuple)):
        data_high, data_low = data
    else:
        data_high, data_low = data, None

    num_filter_low = int(num_filter * ratio_out)
    num_filter_high = num_filter - num_filter_low

    rank = 1
    if isinstance(kernel, (list, tuple)):
        rank = max(rank, len(kernel))
    if isinstance(stride, (list, tuple)):
        rank = max(rank, len(stride))
    if isinstance(dilate, (list, tuple)):
        rank = max(rank, len(dilate))
    kernel = _make_tuple(kernel, rank)
    stride = _make_tuple(stride, rank)
    dilate = _make_tuple(dilate, rank)
    octave = _make_tuple(octave, rank)
    pad = tuple([k // 2 * d for k, d in zip(kernel, dilate)])

    def _init_conv(_data, _num_filter, _suffix):
        return mx.symbol.Convolution(
            data=_data,
            kernel=kernel,
            stride=stride,
            dilate=dilate,
            pad=pad,
            num_filter=_num_filter,
            num_group=num_group,
            workspace=workspace,
            no_bias=no_bias,
            cudnn_tune=cudnn_tune,
            cudnn_off=cudnn_off,
            layout=layout,
            name=_make_name(name, _suffix),
            attr=attr,
            **kwargs
        )

    outputs_high = None
    if num_filter_high > 0:
        outputs_high = _init_conv(data_high, num_filter_high, 'hh')
        if data_low is not None:
            outputs_high = mx.symbol.elemwise_add(
                outputs_high,
                mx.symbol.UpSampling(
                    _init_conv(data_low, num_filter_high, 'lh'),
                    scale=octave[0],
                    sample_type='nearest',
                    name=_make_name(name, 'up'),
                ),
                name=_make_name(name, 'add-h'),
            )

    outputs_low = None
    if num_filter_low > 0:
        outputs_low = _init_conv(
            mx.symbol.Pooling(
                data_high,
                kernel=octave,
                pool_type='avg',
                stride=octave,
                cudnn_off=cudnn_off,
                name=_make_name(name, 'pooling'),
            ),
            num_filter_low, 'hl',
        )
        if data_low is not None:
            outputs_low = mx.symbol.elemwise_add(
                _init_conv(data_low, num_filter_low, 'll'),
                outputs_low,
                name=_make_name(name, 'add-l'),
            )

    if outputs_high is None:
        return outputs_low
    if outputs_low is None:
        return outputs_high
    return outputs_high, outputs_low


def octave_dual(data, builder):
    if not isinstance(data, (list, tuple)):
        data = [data]
    outputs = [builder(datum) for datum in data]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def octave_residual(x, y):
    return octave_dual(list(zip(x, y)), lambda z: z[0] + z[1])
