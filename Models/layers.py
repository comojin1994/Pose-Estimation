from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    ZeroPadding2D,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Add,
    UpSampling2D,
    Concatenate
)


def zeroPadding2D(layer):
    return ZeroPadding2D(layer)


def conv2D(filters, size, strides, padding, batch_norm):
    return Conv2D(filters=filters,
                  kernel_size=size,
                  strides=strides,
                  padding=padding,
                  use_bias=not batch_norm,
                  kernel_regularizer=l2(0.0005)
                  )


def batchNormalization():
    return BatchNormalization()


def leakyRelu(alpha):
    return LeakyReLU(alpha)


def add():
    return Add()


def upsampling2D(size):
    return UpSampling2D(size)


def concatenate():
    return Concatenate()