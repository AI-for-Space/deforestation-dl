
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import Model, load_model

def double_conv_block(x, n_filters):
        # Conv2D then ReLU activation
        conv = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        # Conv2D then ReLU activation
        conv = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(conv)
        return conv

def downsample_block(x, n_filters):
    conv_block = double_conv_block(x, n_filters)
    pool = MaxPooling2D(pool_size=(2, 2),strides=2, padding='same')(conv_block)
    output = Dropout(0.3)(pool)
    return conv_block, output

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = Concatenate(axis=3)([x, conv_features])
    # dropout
    x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x

def unet_model(input_shape):

    inputs = keras.Input(shape = input_shape)
    # encoder: contracting path - downsample
    # 1 - downsample
    conv1, output_conv1 = downsample_block(inputs, 64)
    # 2 - downsample
    conv2, output_conv2 = downsample_block(output_conv1, 128)
    # 3 - downsample
    conv3, output_conv3 = downsample_block(output_conv2, 256)
    # 4 - downsample
    conv4, output_conv4 = downsample_block(output_conv3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(output_conv4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    up_conv6 = upsample_block(bottleneck, conv4, 512)
    # 7 - upsample
    up_conv7 = upsample_block(up_conv6, conv3, 256)
    # 8 - upsample
    up_conv8 = upsample_block(up_conv7, conv2, 128)
    # 9 - upsample
    up_conv9 = upsample_block(up_conv8, conv1, 64)
    # outputs
    outputs = Conv2D(1, 1, padding="same", activation = "sigmoid")(up_conv9)
    # unet model with Keras Functional API
    unet_model = keras.Model(inputs, outputs, name="U-Net")
    unet_model.summary()

    return unet_model
