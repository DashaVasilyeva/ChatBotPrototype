import tensorflow as tf
import numpy as np


from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers import Embedding, Dropout, LayerNormalization


class PositionalEncoding(object):
    def __init__(self, position, d):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self._encoding = np.concatenate([sines, cosines], axis=-1)
        self._encoding = self._encoding[np.newaxis, ...]

    def _get_angles(self, position, i, d):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d))
        return position * angle_rates

    def get_positional_encoding(self):
        return tf.cast(self._encoding, dtype=tf.float32)


class PreProcessingLayer(Layer):
    def __init__(self, neurons_number, vocabulary_size):
        super(PreProcessingLayer, self).__init__()

        # Initialize
        self.neurons_number = neurons_number

        # Add embeddings and positional encoding
        self.embedding = Embedding(vocabulary_size, self.neurons_number)
        positional_encoding_handler = PositionalEncoding(vocabulary_size, self.neurons_number)
        self.positional_encoding = positional_encoding_handler.get_positional_encoding()

        # Add training dropout
        self.dropout = Dropout(0.1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'neurons_number': self.neurons_number,
            'embedding': self.embedding,
            'positional_encoding': self.positional_encoding,
        })
        return config


    def call(self, sequence):
        sequence_length = tf.shape(sequence)[1]
        sequence = self.embedding(sequence)

        sequence *= tf.math.sqrt(tf.cast(self.neurons_number, tf.float32))
        sequence += self.positional_encoding[:, :sequence_length, :]
        sequence = self.dropout(sequence)

        return sequence


class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'projection_dim': self.projection_dim,
            'query_dense': self.query_dense,
            'key_dense': self.key_dense,
            'value_dense': self.value_dense,
            'combine_heads': self.combine_heads,
        })
        return config


    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights


    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(
            query, batch_size
        )
        key = self.separate_heads(
            key, batch_size
        )
        value = self.separate_heads(
            value, batch_size
        )
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(
            concat_attention
        )
        return output


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)


    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config


    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
