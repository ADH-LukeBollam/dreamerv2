import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tfpl = tfp.layers
tfd = tfp.distributions
tfkl = tf.keras.layers
tfb = tfp.bijectors


class SetPrior(tf.keras.Model):
    def __init__(self, event_size, *args, **kwargs):
        super(SetPrior, self).__init__()
        self.event_size = event_size
        mvnd_input_size = 2         # size 2 because loc and scale inputs

        self.parametrization = tfpl.VariableLayer([self.event_size, mvnd_input_size],
                                                  name='loc', dtype=tf.float32)

    def call(self, features):
        # doesnt matter what we pass in here as tf.VariableLayer ignores input (an error gets thrown if empty though)
        batch_size = tf.reduce_sum(tf.shape(features)[:-1])
        params = self.parametrization(None)
        tiled = tf.tile(tf.expand_dims(params, 0), [batch_size, 1, 1])

        mean = tf.reshape(tiled, tf.concat([tf.shape(features)[:-1], self.event_size], 0))
        return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))
        return samples

if __name__ == '__main__':
    batch_size = 1000
    event_size = 2
    # set_sizes = [109, 85, 73, 100, 124, 151]

    prior = SetPrior(event_size)
    distribution = prior(batch_size)
    sample = distribution.sample()

    plt.scatter(sample[..., 0], sample[..., 1])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
