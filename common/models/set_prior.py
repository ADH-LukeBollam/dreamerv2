import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec


class SetPrior(tf.keras.Model):
    def __init__(self, event_size, *args, **kwargs):
        super(SetPrior, self).__init__()
        self.event_size = event_size
        self.parametrization = tf.Variable(tf.zeros(self.event_size, dtype=tf.float32), dtype=tf.float32)

    def call(self, batch_shape):
        batch_rank = tf.size(batch_shape)

        out_rank = tf.concat([tf.ones((batch_rank,), tf.int32), (self.event_size,)], 0)
        out_shape = tf.reshape(self.parametrization, out_rank)
        out = tf.tile(out_shape, tf.concat([batch_shape, (1,)], 0))

        out = tf.cast(out, prec.global_policy().compute_dtype)

        dist = tfd.Normal(out, 1.0)
        return tfd.Independent(dist, 1)

if __name__ == '__main__':
    ev_size = 62
    features = tf.zeros([5, 8, ev_size])
    # set_sizes = [109, 85, 73, 100, 124, 151]

    prior = SetPrior(ev_size)
    distribution = prior(tf.shape(features)[:-1])
    sample = distribution.sample()

    plt.scatter(sample[..., 0], sample[..., 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
