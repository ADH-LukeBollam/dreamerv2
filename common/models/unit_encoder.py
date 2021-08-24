import tensorflow as tf

import common
from models.set_prior import SetPrior
from models.size_predictor import SizePredictor
from models.set_transformer import TransformerLayer, PoolingMultiheadAttention
from pysc2.lib.units import get_unit_embed_lookup
from tensorflow.keras.mixed_precision import experimental as prec
from tensorflow_probability import distributions as tfd


class UnitProcessing(tf.keras.layers.Layer):
    def __init__(self, preprocesing_dim, out_dim):
        super(UnitProcessing, self).__init__()
        self.preprocesing_dim = preprocesing_dim
        self.out_dim = out_dim

        self.conv1 = tf.keras.layers.Conv1D(preprocesing_dim, 1, kernel_initializer='glorot_uniform', use_bias=True)
        self.conv2 = tf.keras.layers.Conv1D(out_dim, 1, kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, unit_feats):
        x = self.conv1(unit_feats)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)

        return x


class UnitEncoder(common.Module):
    def __init__(self, unit_embedding_dim=16, preprocesing_dim=64, num_layers=2, trans_dim=256, num_heads=4):
        super(UnitEncoder, self).__init__()

        num_unit_types = len(set(get_unit_embed_lookup().values()))
        self.type_embedding = tf.keras.layers.Embedding(num_unit_types, unit_embedding_dim)

        self.pointwise_processing = UnitProcessing(preprocesing_dim, trans_dim)

        self.num_layers = num_layers
        self.transformer = [TransformerLayer(trans_dim, num_heads) for _ in range(num_layers)]
        self.transformer_pooling = PoolingMultiheadAttention(trans_dim, 1, 1)

    @tf.function
    def __call__(self, sets):
        x_feats = self.embed_unit_type(sets)

        # flatten our batch + timestep dimensions together
        x = tf.reshape(x_feats, (-1,) + tuple(x_feats.shape[-2:]))

        # get the transformer mask []
        seq = tf.cast(tf.math.equal(x[:, :, 0], 0), tf.float32)
        masked_values = seq[:, tf.newaxis, tf.newaxis, :]

        x = self.pointwise_processing(x)

        for i in range(self.num_layers):
            x = self.transformer[i](x, x, masked_values)

        merged = self.transformer_pooling(x, masked_values)

        # recover our batch and timestep dims
        shape = tf.concat([tf.shape(x_feats)[:-2], [merged.shape[-1]]], 0)
        merged = tf.reshape(merged, shape)

        return merged

    @tf.function
    def embed_unit_type(self, units):
        unit_types = units[..., 0]
        embedded_types = self.type_embedding(unit_types)
        remaining_features = units[..., 1:]
        embedded = tf.concat([embedded_types, remaining_features], axis=-1)

        return embedded

class UnitDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, trans_dim, trans_heads, unit_features):
        super(UnitDecoder, self).__init__()
        # process initial set to transformer dimension
        self.embedding = tf.keras.layers.Conv1D(trans_dim, 1, kernel_initializer='glorot_uniform', use_bias=True,
                                                bias_initializer=tf.constant_initializer(0.1))\

        self.num_layers = num_layers
        self.trans_dim = trans_dim
        self.transformer = [TransformerLayer(trans_dim, trans_heads) for _ in range(num_layers)]

        num_unit_types = len(set(get_unit_embed_lookup().keys()))
        self.out_dense = tf.keras.layers.Dense(num_unit_types + unit_features)

    def call(self, initial_set, encoding, sizes):
        # flatten our batch + timestep dimensions together
        x = tf.reshape(initial_set, (-1,) + tuple(initial_set.shape[-2:]))
        x_encoding = tf.reshape(encoding, (-1, 1) + tuple(encoding.shape[-1:]))

        set_size = tf.shape(x)[1]     # batch, set, features

        # concat the encoding vector onto each initial set element
        encoded_shaped = tf.tile(x_encoding, [1, set_size, 1])
        conditioned_initial_set = tf.concat([x, encoded_shaped], 2)

        mask = tf.reshape(tf.cast(tf.math.logical_not(tf.sequence_mask(sizes, set_size)), tf.float32), [-1, 1, 1, set_size])

        x = self.embedding(conditioned_initial_set)

        for i in range(self.num_layers):
            x = self.transformer[i](x, x, mask)

        out = self.out_dense(x)

        mean = tf.reshape(out, tf.concat([tf.shape(initial_set)[:-1], [tf.shape(out)[-1]]], 0))
        pred_set = tfd.Independent(tfd.Normal(mean, 1), 1)

        return pred_set
