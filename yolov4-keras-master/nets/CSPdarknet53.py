from functools import wraps
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


# --------------------------------------------------#
#   Single convolution
# --------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + Mish
# ---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


# ---------------------------------------------------#
#   CSPdarknet structure block
# ---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    # Perform length and width compression
    preconv1 = ZeroPadding2D(((1, 0), (1, 0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3, 3), strides=(2, 2))(preconv1)

    shortconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)

    # Convolution of the backbone
    mainconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(preconv1)
    # 1x1 convolution->3x3convolution Extract features
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Mish(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (3, 3)))(mainconv)
        mainconv = Add()([mainconv, y])
    # Stacking with residual edge after 1x1 convolution
    postconv = DarknetConv2D_BN_Mish(num_filters // 2 if all_narrow else num_filters, (1, 1))(mainconv)
    route = Concatenate()([postconv, shortconv])

    # the number of channels is integrated
    return DarknetConv2D_BN_Mish(num_filters, (1, 1))(route)


# ---------------------------------------------------#
#   darknet53 body
# ---------------------------------------------------#
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3, 3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3
