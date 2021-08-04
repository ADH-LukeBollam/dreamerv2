import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions

d = tfd.Blockwise(
    [
        tfd.Independent(
            tfd.Normal(
                loc=tf.zeros(4, dtype=tf.float64),
                scale=1),
            reinterpreted_batch_ndims=1),
        tfd.MultivariateNormalTriL(
            scale_tril=tf.eye(2, dtype=tf.float32)),
    ],
    dtype_override=tf.float32,
)
x = d.sample([2, 1])
y = d.log_prob(x)
x.shape  # ==> (2, 1, 4 + 2)
x.dtype  # ==> tf.float32
y.shape  # ==> (2, 1)
y.dtype  # ==> tf.float32

d.mean()  # ==> np.zeros((4 + 2,))