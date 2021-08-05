import tensorflow as tf

import common
from models.set_prior import SetPrior
from models.size_predictor import SizePredictor
from models.set_transformer import TransformerLayer, PoolingMultiheadAttention
from pysc2.lib.units import get_unit_embed_lookup
from tensorflow.keras.mixed_precision import experimental as prec


class UnitProcessing(tf.keras.layers.Layer):
    def __init__(self, unit_embedding_dim, preprocesing_dim, out_dim):
        super(UnitProcessing, self).__init__()
        self.preprocesing_dim = preprocesing_dim
        self.out_dim = out_dim
        self.unit_embedding_dim = unit_embedding_dim

        num_unit_types = len(set(get_unit_embed_lookup().values()))
        self.type_embedding = tf.keras.layers.Embedding(num_unit_types, unit_embedding_dim)

        self.conv1 = tf.keras.layers.Conv1D(preprocesing_dim, 1, kernel_initializer='glorot_uniform', use_bias=True)
        self.conv2 = tf.keras.layers.Conv1D(out_dim, 1, kernel_initializer='glorot_uniform', use_bias=True)

    def call(self, units):
        unit_types = units[:, :, 0]
        embedded_types = self.type_embedding(unit_types)
        remaining_features = units[:, :, 1:]
        embedded = tf.concat([embedded_types, remaining_features], axis=-1)

        x = self.conv1(embedded)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)

        return x


class UnitEncoder(common.Module):
    def __init__(self, unit_embedding_dim=16, preprocesing_dim=64, num_layers=2, trans_dim=256, num_heads=4):
        super(UnitEncoder, self).__init__()

        self.pointwise_processing = UnitProcessing(unit_embedding_dim, preprocesing_dim, trans_dim)

        self.num_layers = num_layers
        self.transformer = [TransformerLayer(trans_dim, num_heads) for _ in range(num_layers)]
        self.transformer_pooling = PoolingMultiheadAttention(trans_dim, 1, 1)

    @tf.function
    def __call__(self, sets):
        # flatten our batch + timestep dimensions together
        x = tf.reshape(sets, (-1,) + tuple(sets.shape[-2:]))

        # get the transformer mask []
        seq = tf.cast(tf.math.equal(x[:, :, 0], 0), tf.float32)
        masked_values = seq[:, tf.newaxis, tf.newaxis, :]

        x = self.pointwise_processing(x)

        for i in range(self.num_layers):
            x = self.transformer[i](x, x, masked_values)

        merged = self.transformer_pooling(x, masked_values)

        # recover our batch and timestep dims
        shape = tf.concat([tf.shape(sets)[:-2], [merged.shape[-1]]], 0)
        merged = tf.reshape(merged, shape)

        return merged


class SetDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, trans_dim, num_heads):
        super(SetDecoder, self).__init__()
        # process initial set to transformer dimension
        self.embedding = tf.keras.layers.Conv1D(trans_dim, 1, kernel_initializer='glorot_uniform', use_bias=True,
                                                bias_initializer=tf.constant_initializer(0.1))\

        self.num_layers = num_layers
        self.transformer = [TransformerLayer(trans_dim, num_heads) for _ in range(num_layers)]

    def call(self, initial_set, mask):
        x = self.embedding(initial_set)

        for i in range(self.num_layers):
            x = self.transformer[i](x, x, mask)

        return x


class SetDecoder(tf.keras.Model):
    def __init__(self, encoder_latent, transformer_layers, transformer_dim,
                 transformer_num_heads, num_element_features, size_pred_width, pad_value, max_set_size):
        super(SetDecoder, self).__init__()

        self.pad_value = pad_value
        self.max_set_size = max_set_size
        self.num_element_features = num_element_features

        self._prior = SetPrior(num_element_features)

        self._encoder = UnitEncoder(encoder_latent, transformer_layers, transformer_dim, transformer_num_heads)
        self._decoder = SetDecoder(transformer_layers, transformer_dim, transformer_num_heads)

        # initialise the output to predict points at the center of our canvas
        self._set_prediction = tf.keras.layers.Conv1D(num_element_features, 1, kernel_initializer='zeros',
                                                     bias_initializer=tf.keras.initializers.constant(0.5),
                                                     use_bias=True)

        self._size_predictor = SizePredictor(size_pred_width, max_set_size)

    def call(self, initial_set, sampled_set, sizes):

        # encode the input set
        encoded = self._encoder(initial_set, masked_values)  # pooled: [batch_size, num_features]

        # concat the encoded set vector onto each initial set element
        encoded_shaped = tf.tile(encoded, [1, self.max_set_size, 1])
        sampled_elements_conditioned = tf.concat([sampled_set, encoded_shaped], 2)

        pred_set_latent = self._decoder(sampled_elements_conditioned, masked_values)

        pred_set = self._set_prediction(pred_set_latent)
        return pred_set

    def sample_prior(self, sizes):
        total_elements = tf.reduce_sum(sizes)
        sampled_elements = self._prior(total_elements)  # [batch_size, max_set_size, num_features]
        return sampled_elements

    def sample_prior_batch(self, sizes):
        sampled_elements = self.sample_prior(sizes)
        samples_ragged = tf.RaggedTensor.from_row_lengths(sampled_elements, sizes)
        padded_samples = samples_ragged.to_tensor(default_value=self.pad_value,
                                                  shape=[sizes.shape[0], self.max_set_size, self.num_element_features])
        return padded_samples

    def encode_set(self, initial_set, sizes):
        return self._encoder(initial_set, sizes)

    def predict_size(self, embedding):
        sizes = self._size_predictor(embedding)
        sizes = tf.keras.activations.softmax(sizes, -1)
        return sizes

    def get_autoencoder_weights(self):
        return self._encoder.trainable_weights + \
               self._decoder.trainable_weights + \
               self._set_prediction.trainable_weights

    def get_prior_weights(self):
        return self._prior.trainable_weights

    def get_size_predictor_weights(self):
        return self._size_predictor.trainable_weights
