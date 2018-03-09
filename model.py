import numpy as np
np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.activations import relu
from keras.layers import Input, Dense, Embedding, concatenate, add, merge
from keras.layers import GRU, Bidirectional, LSTM
from keras.layers import Activation, Conv1D, Flatten, Lambda
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import SpatialDropout1D
from keras.callbacks import Callback
from keras import regularizers

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

from keras import backend as K
from keras.engine.topology import Layer
#from keras import initializations
from keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim

def roc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

'''
CNN
'''
def get_CNN_model(maxlen, max_features, embed_size, embedding_matrix,
                  dropout, num_filter, kernel_size, reg):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Conv1D(num_filter, kernel_size, activation='relu',
                kernel_regularizer=regularizers.l2(reg))(x)
    x = GlobalMaxPooling1D()(x)
    outp = Dense(6, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])
    return model

def get_CNN_model_by_cat(maxlen, max_features, embed_size, embedding_matrix,
                         dropout, num_filter, kernel_size, reg):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Conv1D(num_filter, kernel_size, activation='relu',
                kernel_regularizer=regularizers.l2(reg))(x)
    x = GlobalMaxPooling1D()(x)
    outp = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])
    return model

def get_multiChannel_CNN_model(maxlen, max_features, embed_size, embedding_matrix,
                               dropout, num_filter, kernel_size, reg):
    inp1 = Input(shape=(maxlen,))
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp1)
    x1 = SpatialDropout1D(dropout)(x1)
    x1 = Conv1D(num_filter, kernel_size, activation='relu',
                kernel_regularizer=regularizers.l2(reg))(x1)
    x1 = GlobalMaxPooling1D()(x1)

    inp2 = Input(shape=(maxlen,))
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp2)
    x2 = SpatialDropout1D(dropout)(x2)
    x2 = Conv1D(num_filter, kernel_size, activation='relu',
                kernel_regularizer=regularizers.l2(reg))(x2)
    x2 = GlobalMaxPooling1D()(x2)
    x2 = Flatten()(x2)

    merged = concatenate([x1, x2])
    outp = Dense(6, activation='sigmoid')(merged)
    
    model = Model(inputs=[inp1, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])
    return model

def get_multiChannel_CNN_model_by_cat(maxlen, max_features, embed_size, embedding_matrix,
                                      dropout, num_filter, kernel_size, reg):
    inp1 = Input(shape=(maxlen,))
    x1 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp1)
    x1 = SpatialDropout1D(dropout)(x1)
    x1 = Conv1D(num_filter, kernel_size, activation='relu',
                kernel_regularizer=regularizers.l2(reg))(x1)
    x1 = GlobalMaxPooling1D()(x1)

    inp2 = Input(shape=(maxlen,))
    x2 = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp2)
    x2 = SpatialDropout1D(dropout)(x2)
    x2 = Conv1D(num_filter, kernel_size, activation='relu',
                kernel_regularizer=regularizers.l2(reg))(x2)
    x2 = GlobalMaxPooling1D()(x2)
    x2 = Flatten()(x2)

    merged = concatenate([x1, x2])
    outp = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=[inp1, inp2], outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])
    return model

def get_dpcnn_model(maxlen, max_features, embed_size, embedding_matrix,
                    dropout, num_filter, kernel_size, reg):
    inp = Input(shape=(maxlen, ))
    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    embedding = SpatialDropout1D(dropout)(embedding)
    embedding = Activation('relu')(embedding)  # pre activation
    block1 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(embedding)
    block1 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(block1)
    # reshape layer if needed
    conc1 = None
    if num_filter != embedding_size:
        embedding_resize = Conv1D(num_filter, kernel_size=1, padding='same', activation='linear',
                                kernel_regularizer=regularizers.l2(reg))(embedding)
        block1 = Lambda(relu)(block1)
        conc1 = add([embedding_resize, block1])
    else:
        conc1 = add([embedding, block1])
    
    # block 2 & block 3 are dpcnn repeating blocks
    downsample1 = MaxPooling1D(pool_size=3, strides=2, padding='valid')(conc1)
    block2 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(downsample1)
    block2 = SpatialDropout1D(dropout)(block2)
    block2 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(block2)
    block2 = SpatialDropout1D(dropout)(block2)
    conc2 = add([downsample1, block2])
    
    downsample2 = MaxPooling1D(pool_size=3, strides=2, padding='valid')(conc2)
    block3 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(downsample2)
    block3 = SpatialDropout1D(dropout)(block3)
    block3 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(block3)
    block3 = SpatialDropout1D(dropout)(block3)
    conc3 = add([downsample2, block3])
    
    after_pool = MaxPooling1D(pool_size=pool_size, strides=num_strides)(conc3)
    after_pool = Flatten()(after_pool)
    outp = Dense(6, activation="sigmoid")(after_pool)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])

    return model

def get_dpcnn_model_by_cat(maxlen, max_features, embed_size, embedding_matrix,
                           dropout, num_filter, kernel_size, reg):
    inp = Input(shape=(maxlen, ))
    embedding = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    embedding = SpatialDropout1D(dropout)(embedding)
    embedding = Activation('relu')(embedding)  # pre activation
    block1 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(embedding)
    block1 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(block1)
    # reshape layer if needed
    conc1 = None
    if num_filter != embedding_size:
        embedding_resize = Conv1D(num_filter, kernel_size=1, padding='same', activation='linear',
                                kernel_regularizer=regularizers.l2(reg))(embedding)
        block1 = Lambda(relu)(block1)
        conc1 = add([embedding_resize, block1])
    else:
        conc1 = add([embedding, block1])
    
    # block 2 & block 3 are dpcnn repeating blocks
    downsample1 = MaxPooling1D(pool_size=3, strides=2, padding='valid')(conc1)
    block2 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(downsample1)
    block2 = SpatialDropout1D(dropout)(block2)
    block2 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(block2)
    block2 = SpatialDropout1D(dropout)(block2)
    conc2 = add([downsample1, block2])
    
    downsample2 = MaxPooling1D(pool_size=3, strides=2, padding='valid')(conc2)
    block3 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(downsample2)
    block3 = SpatialDropout1D(dropout)(block3)
    block3 = Conv1D(num_filter, kernel_size, padding='same',
                    kernel_regularizer=regularizers.l2(reg))(block3)
    block3 = SpatialDropout1D(dropout)(block3)
    conc3 = add([downsample2, block3])
    
    after_pool = MaxPooling1D(pool_size=pool_size, strides=num_strides)(conc3)
    after_pool = Flatten()(after_pool)
    outp = Dense(1, activation="sigmoid")(after_pool)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])

    return model

'''
LSTM
'''
def get_LSTM_pool_model(maxlen, max_features, embed_size, embedding_matrix,
                        dropout, units, reg):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(reg)))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])
    return model

def get_LSTM_attention_model(maxlen, max_features, embed_size, embedding_matrix, \
                             dropout, units, reg):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(LSTM(units, return_sequences=True, kernel_regularizer=regularizers.l2(reg)))(x)
    x = Attention(maxlen)(x)
    outp = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])
    return model


'''
GRU
'''
def get_GRU_pool_model(maxlen, max_features, embed_size, embedding_matrix,
                       dropout, units, reg):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(GRU(units, return_sequences=True, kernel_regularizer=regularizers.l2(reg)))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])

    return model

def get_GRU_attention_model(maxlen, max_features, embed_size, embedding_matrix,
                            dropout, units, reg):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout)(x)
    x = Bidirectional(GRU(units, return_sequences=True, kernel_regularizer=regularizers.l2(reg)))(x)
    x = Attention(maxlen)(x)
    outp = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', roc])

    return model