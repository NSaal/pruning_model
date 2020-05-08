from tensorflow.python.keras import backend
from tensorflow_model_optimization.sparsity import keras as sparsity
import inspect
import os

import numpy as np
import tensorflow as tf
import time
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers
from models.official.vision.image_classification.resnet import imagenet_preprocessing

VGG_MEAN = [103.939, 116.779, 123.68]

layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
  return regularizers.l2(l2_weight_decay) if use_l2_regularizer else None


def vgg16(num_classes,
          batch_size=None,
          use_l2_regularizer=True,
          rescale_inputs=False,
          batch_norm_decay=0.9,
          batch_norm_epsilon=1e-5):

    input_shape = (28, 28, 1)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    if rescale_inputs:
        # Hub image modules expect inputs in the range [0, 1]. This rescales these
        # inputs to the range expected by the trained model.
        x = layers.Lambda(
            lambda x: x * 255.0 - backend.constant(
                imagenet_preprocessing.CHANNEL_MEANS,
                shape=[1, 1, 3],
                dtype=x.dtype),
            name='rescale')(
                img_input)
    else:
        x = img_input
    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    x = layers.Conv2D(
        64, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1_1')(
        x)
    x = layers.Conv2D(
        64, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1_2')(
        x)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same', name='pool1')(x)
    
    x = layers.Conv2D(
        128, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv2_1')(
        x)
    x = layers.Conv2D(
        128, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv2_2')(
        x)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same', name='pool2')(x)

    x = layers.Conv2D(
        256, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv3_1')(
        x)
    x = layers.Conv2D(
        256, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv3_2')(
        x)
    x = layers.Conv2D(
        256, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv3_3')(
        x)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same', name='pool3')(x)

    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv4_1')(
        x)
    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv4_2')(
        x)
    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv4_3')(
        x)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same', name='pool4')(x)

    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv5_1')(
        x)
    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv5_2')(
        x)
    x = layers.Conv2D(
        512, (3, 3),
        strides=(1, 1),
        padding='SAME',
        activation='relu',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv5_3')(
        x)
    x = layers.MaxPooling2D((3, 3), strides=(
        2, 2), padding='same', name='pool5')(x)

    x = layers.Dense(4096, activation='relu', name='fc1')(x)
    x = layers.Dense(4096, activation='relu', name='fc2')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='relu', name='fc3')(x)
    #x = layers.Softmax()(x)
    x = layers.Activation('softmax', dtype='float32')(x)
    
    return models.Model(img_input, x, name='vgg16')
