import tensorflow as tf
from tensorflow_probability import distributions as tfd


if __name__ == '__main__':
    # # this works
    # x = tf.constant([[0.5, 0.1, 0.35], [0.4, 0.5, 0.8]], tf.float32)
    #
    # p_x = tfd.Independent(tfd.Normal(x, 1), 0)
    #
    # transposed_elems = tf.expand_dims(tf.transpose(x, perm=(1, 0)), axis=-1)
    # pointwise_log_prob = p_x.log_prob(transposed_elems)

    # this does not work
    x = tf.constant([[[0.5, 0.75], [0.1, 0.25], [0.35, 0.9]], [[0.4, 0.45], [0.5, 0.7], [0.8, 0.25]]], tf.float32)
    p_x = tfd.Independent(tfd.Normal(x, 1), 1)

    transposed_elems = tf.expand_dims(tf.transpose(x, perm=(1, 0, 2)), axis=-2)
    pointwise_log_prob = p_x.log_prob(transposed_elems)
    pass
