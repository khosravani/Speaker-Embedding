from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import *
from keras import losses as Loss
from keras.optimizers import *
from keras import regularizers
from keras import metrics, losses
from keras import backend as K
from keras.initializers import random_normal
from keras.activations import relu, softmax
from keras.regularizers import l2
from keras.utils.generic_utils import get_custom_objects

from keras.engine.base_layer import Layer
from keras import activations

import numpy as np

# Triplet loss layer
class TripletLossLayer(Layer):
    def __init__(self, batch_spk_samples=8, batch_size=128, **kwargs):
        self.is_placeholder = True
        self.batch_size = batch_size
        self.batch_spk_samples = batch_spk_samples
        super(TripletLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.imposter = self.add_weight(
            name='imposter', 
            shape=(256, input_shape[1][-1]),
            initializer='normal',
            trainable=False)
        
        self.w = self.add_weight(
            name='w', 
            shape=(1,),
            initializer='normal',
            trainable=True)
        
        self.b = self.add_weight(
            name='b', 
            shape=(1,),
            initializer='normal',
            trainable=True)

        super(TripletLossLayer, self).build(input_shape)

    def triplet_loss(self, spks, embeds):
        similarity = K.dot(embeds, K.transpose(embeds))
        similarity = self.w * similarity + self.b
        similarity = 1. / (1. + K.exp(-similarity))

        y_pos = K.dot(spks, K.transpose(spks))
        y_neg = 1. - y_pos
        y_pos = y_pos - K.eye(self.batch_size)

        pos = similarity * y_pos
        pos = K.sum(pos, axis=-1, keepdims=True) / (self.batch_spk_samples - 1.)

        neg = similarity * y_neg
        nontar = K.cast(K.greater(neg, pos - 0.1), dtype=K.dtype(embeds))
        neg = nontar * neg
        nontar = K.sum(nontar, axis=-1, keepdims=True)
        neg = K.sum(neg, axis=-1, keepdims=True) / (nontar + K.epsilon())
        
        return K.mean(neg - pos + 1.)

    def call(self, inputs):
        spks = inputs[0]
        embeds = inputs[1]
        loss = self.triplet_loss(spks, embeds)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return embeds

class TDNNLayer(Layer):
    """TDNNLayer
    TDNNLayer sounds like 1D conv with extra steps. Why not doing it with Keras ?
    This layer inherits the Layer class from Keras and is inspired by conv1D layer.
    The documentation will be added later.
    """

    def __init__(self,
                 input_context=[-2, 2],
                 sub_sampling=False,
                 initializer='uniform',
                 activation=None,
                 **kwargs):

        self.input_context = input_context
        self.sub_sampling = sub_sampling
        self.initializer = initializer
        self.activation = activations.get(activation)
        super(TDNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        kernel_shape = (self.input_context[1] - self.input_context[0] + 1, 1)
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.initializer,
                                      trainable=True)
        self.mask = np.zeros(kernel_shape)
        self.mask[0][0] = 1
        self.mask[self.input_context[1]-self.input_context[0]][0] = 1

        if self.sub_sampling:
            self.kernel = self.kernel * self.mask

        super(TDNNLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        if self.sub_sampling:
            output = K.conv1D(inputs,
                              self.kernel,
                              stride=1,
                              padding=0,
                              )
        else:
            output = K.conv1D(inputs,
                              self.kernel * self.mask,
                              stride=1,
                              padding=0,
                              )
        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]-self.input_context[1] + self.input_context[0]


def CNN_Kaldi(
    input_dim=64,
    batch_nspks=16,
    batch_spk_samples=8,
    output_dim=200):

    def lengthnorm(embed):
        return K.l2_normalize(embed, axis=-1)

    def lambda_average(x):
        return K.mean(x, axis=1)

    # input layer
    x_input = Input(shape=(None, input_dim), name='features')
    spks_input = Input(shape=(batch_nspks,), dtype='float32', name='labels')

    y = Reshape(((-1, input_dim, 1)))(x_input)

    y = Conv2D(32, (5, 5), name='cnn1', padding='same', strides=(1, 1))(y)
    y = Activation('relu')(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dropout(0.0)(y)
    y = MaxPooling2D(pool_size=(2, 1))(y)

    y = Conv2D(32, (5, 5), name='cnn2', padding='same', strides=(1, 1))(y)
    y = Activation('relu')(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dropout(0.0)(y)
    y = MaxPooling2D(pool_size=(2, 1))(y)

    y = Reshape(((-1, input_dim * 32)))(y)

    y = TimeDistributed(Dense(400, name='fc1'))(y)
    y = Activation('relu')(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dropout(0.0)(y)

    y = TimeDistributed(Dense(400, name='fc2'))(y)
    y = Activation('relu')(y)
    y = BatchNormalization(axis=-1)(y)
    y = Dropout(0.0)(y)

    ya = Bidirectional(LSTM(
        200,
        dropout=0.0, recurrent_dropout=0.0, return_sequences=False, 
        name='lstm', trainable=True), merge_mode='concat', name='blstm')(y)
    y = Activation('relu')(ya)
    y = BatchNormalization(axis=-1, name='bn2')(y)
    y = Dropout(0.0)(y)

    yb = Dense(400, name='fc3')(y)
    y = Activation('relu')(yb)
    y = BatchNormalization(axis=-1)(y)
    y = Dropout(0.0)(y)

    z = Dense(400, activation='linear', name='pca')(y)
    embed = Lambda(lengthnorm, name='length_norm')(z)
    
    output = TripletLossLayer(  
        batch_spk_samples=batch_spk_samples,
        batch_size=batch_nspks * batch_spk_samples,
        name='triplet_loss')([spks_input, embed])

    model = Model(inputs=[x_input, spks_input], outputs=[output])
    model.summary()

    generator = Model(inputs=[x_input], outputs=[embed])

    return model, generator

