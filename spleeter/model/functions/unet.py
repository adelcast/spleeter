#!/usr/bin/env python
# coding: utf8

"""
This module contains building functions for U-net source
separation models in a similar way as in A. Jansson et al. "Singing
voice separation with deep u-net convolutional networks", ISMIR 2017.
Each instrument is modeled by a single U-net convolutional
/ deconvolutional network that take a mix spectrogram as input and the
estimated sound spectrogram as output.
"""

import json
from functools import partial

# pylint: disable=import-error
import tensorflow as tf

from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    ELU,
    LeakyReLU,
    Multiply,
    ReLU,
    Softmax,
    Input,
    Lambda)
from tensorflow.keras.models import Model

from tensorflow.compat.v1 import logging
from tensorflow.compat.v1.keras.initializers import he_uniform
# pylint: enable=import-error

from ...utils.tensor import pad_and_partition, pad_and_reshape
from . import apply

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


def _get_conv_activation_layer(params):
    """

    :param params:
    :returns: Required Activation function.
    """
    conv_activation = params.get('conv_activation')
    if conv_activation == 'ReLU':
        return ReLU()
    elif conv_activation == 'ELU':
        return ELU()
    return LeakyReLU(0.2)


def _get_deconv_activation_layer(params):
    """

    :param params:
    :returns: Required Activation function.
    """
    deconv_activation = params.get('deconv_activation')
    if deconv_activation == 'LeakyReLU':
        return LeakyReLU(0.2)
    elif deconv_activation == 'ELU':
        return ELU()
    return ReLU()

class Unet(tf.keras.layers.Layer):
    def __init__(self):
        self.conv1 = None
        self.batch1 = None
        self.rel1 = None

        self.conv2 = None
        self.batch2 = None
        self.rel2 = None

        self.conv3 = None
        self.batch3 = None
        self.rel3 = None

        self.conv4 = None
        self.batch4 = None
        self.rel4 = None

        self.conv5 = None
        self.batch5 = None
        self.rel5 = None

        self.conv6 = None
        self.batch6 = None
        self._ = None

        self.up1 = None
        self.batch7 = None
        self.drop1 = None
        self.merge1 = None

        self.up2 = None
        self.batch8 = None
        self.drop2 = None
        self.merge2 = None

        self.up3 = None
        self.batch9 = None
        self.drop3 = None
        self.merge3 = None

        self.up4 = None
        self.batch10 = None
        self.merge4 = None

        self.up5 = None
        self.batch11 = None
        self.merge5 = None

        self.up6 = None
        self.batch12 = None

        self.up7 = None

        super(Unet, self).__init__()

    def call(self, spectrogram, params, output_name='output'):
        logging.info(f'Apply unet for {output_name}')
        conv_n_filters = params.get('conv_n_filters', [16, 32, 64, 128, 256, 512])
        conv_activation_layer = _get_conv_activation_layer(params)
        deconv_activation_layer = _get_deconv_activation_layer(params)
        kernel_initializer = he_uniform(seed=50)
        conv2d_factory = partial(
            Conv2D,
            strides=(2, 2),
            padding='same',
            kernel_initializer=kernel_initializer)
        # First layer.
        if self.conv1 is None:
            self.conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(spectrogram)
        if self.batch1 is None:
            self.batch1 = BatchNormalization(axis=-1)(self.conv1)
        if self.rel1 is None:
            self.rel1 = conv_activation_layer(self.batch1)

        # Second layer.
        if self.conv2 is None:
            self.conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(self.rel1)
        if self.batch2 is None:
            self.batch2 = BatchNormalization(axis=-1)(self.conv2)
        if self.rel2 is None:
            self.rel2 = conv_activation_layer(self.batch2)

        # Third layer.
        if self.conv3 is None:
            self.conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(self.rel2)
        if self.batch3 is None:
            self.batch3 = BatchNormalization(axis=-1)(self.conv3)
        if self.rel3 is None:
            self.rel3 = conv_activation_layer(self.batch3)

        # Fourth layer.
        if self.conv4 is None:
            self.conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(self.rel3)
        if self.batch4 is None:
            self.batch4 = BatchNormalization(axis=-1)(self.conv4)
        if self.rel4 is None:
            self.rel4 = conv_activation_layer(self.batch4)

        # Fifth layer.
        if self.conv5 is None:
            self.conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(self.rel4)
        if self.batch5 is None:
            self.batch5 = BatchNormalization(axis=-1)(self.conv5)
        if self.rel5 is None:
            self.rel5 = conv_activation_layer(self.batch5)

        # Sixth layer
        if self.conv6 is None:
            self.conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(self.rel5)
        if self.batch6 is None:
            self.batch6 = BatchNormalization(axis=-1)(self.conv6)
        if self.conv3 is None:
            self._ = conv_activation_layer(self.batch6)

        conv2d_transpose_factory = partial(
            Conv2DTranspose,
            strides=(2, 2),
            padding='same',
            kernel_initializer=kernel_initializer)

        if self.up1 is None:
            self.up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))((self.conv6))
            self.up1 = deconv_activation_layer(self.up1)
        if self.batch7 is None:
            self.batch7 = BatchNormalization(axis=-1)(self.up1)
        if self.drop1 is None:
            self.drop1 = Dropout(0.5)(self.batch7)
        if self.merge1 is None:
            self.merge1 = Concatenate(axis=-1)([self.conv5, self.drop1])
        if self.up2 is None:
            self.up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((self.merge1))
            self.up2 = deconv_activation_layer(self.up2)
        if self.batch8 is None:
            self.batch8 = BatchNormalization(axis=-1)(self.up2)
        if self.drop2 is None:
            self.drop2 = Dropout(0.5)(self.batch8)
        if self.merge2 is None:
            self.merge2 = Concatenate(axis=-1)([self.conv4, self.drop2])
        if self.up3 is None:
            self.up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((self.merge2))
            self.up3 = deconv_activation_layer(self.up3)
        if self.batch9 is None:
            self.batch9 = BatchNormalization(axis=-1)(self.up3)
        if self.drop3 is None:
            self.drop3 = Dropout(0.5)(self.batch9)
        if self.merge3 is None:
            self.merge3 = Concatenate(axis=-1)([self.conv3, self.drop3])
        if self.up4 is None:
            self.up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((self.merge3))
            self.up4 = deconv_activation_layer(self.up4)
        if self.batch10 is None:
            self.batch10 = BatchNormalization(axis=-1)(self.up4)
        if self.merge4 is None:
            self.merge4 = Concatenate(axis=-1)([self.conv2, self.batch10])
        if self.up5 is None:
            self.up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((self.merge4))
            self.up5 = deconv_activation_layer(self.up5)
        if self.batch11 is None:
            self.batch11 = BatchNormalization(axis=-1)(self.up5)
        if self.merge5 is None:
            self.merge5 = Concatenate(axis=-1)([self.conv1, self.batch11])
        if self.up6 is None:
            self.up6 = conv2d_transpose_factory(1, (5, 5), strides=(2, 2))((self.merge5))
            self.up6 = deconv_activation_layer(self.up6)
        if self.batch12 is None:
            self.batch12 = BatchNormalization(axis=-1)(self.up6)
        # Last layer to ensure initial shape reconstruction.

        if self.up7 is None:
            self.up7 = Conv2D(
                2,
                (4, 4),
                dilation_rate=(2, 2),
                activation='sigmoid',
                padding='same',
                kernel_initializer=kernel_initializer)((self.batch12))
        output = Multiply(name=output_name)([self.up7, spectrogram])

        return output

# Keras layer to get Waveforms out of a spectrogram
class Waveform(tf.keras.layers.Layer):
    WINDOW_COMPENSATION_FACTOR = 2./3.
    EPSILON = 1e-10

    def __init__(self):
        super(Waveform, self).__init__()

    def _extend_mask(self, mask, params):
        """ Extend mask, from reduced number of frequency bin to the number of
        frequency bin in the STFT.
        :param mask: restricted mask
        :returns: extended mask
        :raise ValueError: If invalid mask_extension parameter is set.
        """
        extension = params['mask_extension']
        # Extend with average
        # (dispatch according to energy in the processed band)
        if extension == "average":
            extension_row = tf.reduce_mean(mask, axis=2, keepdims=True)
        # Extend with 0
        # (avoid extension artifacts but not conservative separation)
        elif extension == "zeros":
            mask_shape = tf.shape(mask)
            extension_row = tf.zeros((
                mask_shape[0],
                mask_shape[1],
                1,
                mask_shape[-1]))
        else:
            raise ValueError(f'Invalid mask_extension parameter {extension}')
        n_extra_row = (params['frame_length']) // 2 + 1 - params['F']
        extension = tf.tile(extension_row, [1, 1, n_extra_row, 1])
        return tf.concat([mask, extension], axis=2)

    def _inverse_stft(self, stft, params, waveform):
            """ Inverse and reshape the given STFT

            :param stft: input STFT
            :returns: inverse STFT (waveform)
            """
            inversed = tf.signal.inverse_stft(
                tf.transpose(a=stft, perm=[2, 0, 1]),
                params['frame_length'],
                params['frame_step'],
                window_fn=lambda frame_length, dtype: (
                    tf.signal.hann_window(frame_length, periodic=True, dtype=dtype))
            ) * self.WINDOW_COMPENSATION_FACTOR
            reshaped = tf.transpose(a=inversed)
            return reshaped[:tf.shape(input=waveform)[0], :]

    def call(self, stft, spectogram_dict, params, waveform):
        """ Perform ratio mask separation

        :param output_dict: dictionary of estimated spectrogram (key: instrument
            name, value: estimated spectrogram of the instrument)
        :returns: dictionary of separated waveforms (key: instrument name,
            value: estimated waveform of the instrument)
        """
        separation_exponent = params['separation_exponent']
        output_sum = tf.reduce_sum(
            input_tensor=[e ** separation_exponent for e in spectogram_dict.values()],
            axis=0
        ) + self.EPSILON

        output_waveform = {}
        for instrument in ['vocals', 'accompanimient']:
            output = spectogram_dict[f'{instrument}']
            # Compute mask with the model.
            instrument_mask = (
                output ** separation_exponent
                + (self.EPSILON / len(spectogram_dict))) / output_sum
            # Extend mask;
            instrument_mask = self._extend_mask(instrument_mask, params)
            # Stack back mask.
            old_shape = tf.shape(input=instrument_mask)
            new_shape = tf.concat(
                [[old_shape[0] * old_shape[1]], old_shape[2:]],
                axis=0)
            instrument_mask = tf.reshape(instrument_mask, new_shape)
            # Remove padded part (for mask having the same size as STFT);
            stft_feature = stft
            instrument_mask = instrument_mask[
                :tf.shape(input=stft_feature)[0], ...]
            # Compute masked STFT and normalize it.
            output_waveform[instrument] = self._inverse_stft(
                tf.cast(instrument_mask, dtype=tf.complex64) * stft_feature, params, waveform)

        return output_waveform

# Keras layer to get a spectrogram out of a waveform
class Spectrogram(tf.keras.layers.Layer):
    def __init__(self):
        super(Spectrogram, self).__init__()

    def call(self, waveform, params):
        stft_feature = tf.transpose(
            a=tf.signal.stft(
                tf.transpose(a=waveform),
                params['frame_length'],
                params['frame_step'],
                window_fn=lambda frame_length, dtype: (
                    tf.signal.hann_window(frame_length, periodic=True, dtype=dtype)),
                pad_end=True),
            perm=[1, 2, 0])
        spectrogram = tf.abs(
            pad_and_partition(stft_feature, params['T']))[:, :, :params['F'], :]
        return stft_feature, spectrogram

def build_unet(
        waveform,
        output_name='output',
        params={},
        output_mask_logit=False):
    """ Apply a convolutionnal U-net to model a single instrument (one U-net
    is used for each instrument).

    :param input_tensor:
    :param output_name: (Optional) , default to 'output'
    :param params: (Optional) , default to empty dict.
    :param output_mask_logit: (Optional) , default to False.
    """

    # Pre NN - Get spectrogram
    stft, spectrogram = Spectrogram()(waveform, params)

    # TODO: For testing,  just get the vocals out
    vocals = Unet()(spectrogram, params)

    #accompanimient = Unet()(spectrogram, params)
    #spectrogram_dict = {
    #    'vocals': vocals,
    #    'accompanimient': accompanimient
    #}

    # Get waveform out of spectrogram
    #stem_waveform = Waveform()(stft, spectrogram_dict, params, waveform)
    #model = Model(inputs=waveform, outputs=stem_waveform, name='model')

    model = Model(inputs=waveform, outputs=vocals, name='model')
    return model

if __name__ == "__main__":
    # Load model params
    CONFIG_2STEMS = '/home/adelcast/git/spleeter3/configs/2stems/base_config.json'
    with open(CONFIG_2STEMS, 'r') as stream:
        params = json.load(stream)

    # Build input
    shape = (None, params['n_channels'])
    waveform = Input((2))

    # Build NN
    unet_model = build_unet(waveform, params=params)

    # Convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(unet_model)

    # MLIR converter chokes on tf.pad with complex
    # possibly a limitation?
    converter.experimental_new_converter = False

    # THIS IS NEEDED BY NON MLIR backend
    #converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflitemodel = converter.convert()