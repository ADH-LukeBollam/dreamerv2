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
    def __init__(self, preprocesing_dim=64, num_layers=2, trans_dim=256, num_heads=4):
        super(UnitEncoder, self).__init__()

        self.pointwise_processing = UnitProcessing(preprocesing_dim, trans_dim)

        self.num_layers = num_layers
        self.transformer = [TransformerLayer(trans_dim, num_heads) for _ in range(num_layers)]
        self.transformer_pooling = PoolingMultiheadAttention(trans_dim, 1, 1)

    @tf.function
    def __call__(self, unit_feats):
        # flatten our batch + timestep dimensions together
        x = tf.reshape(unit_feats, (-1,) + tuple(unit_feats.shape[-2:]))

        # get the transformer mask []
        seq = tf.cast(tf.math.equal(x[:, :, 0], 0), tf.float32)
        masked_values = seq[:, tf.newaxis, tf.newaxis, :]

        x = self.pointwise_processing(x)

        for i in range(self.num_layers):
            x = self.transformer[i](x, x, masked_values)

        merged = self.transformer_pooling(x, masked_values)

        # recover our batch and timestep dims
        shape = tf.concat([tf.shape(unit_feats)[:-2], [merged.shape[-1]]], 0)
        merged = tf.reshape(merged, shape)

        return merged
