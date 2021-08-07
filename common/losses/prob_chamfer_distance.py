import tensorflow as tf
from tensorflow_probability import distributions as tfd

# this is modified from tensorflows chamfer distance func
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py


# modification of chamfer distance to calculate smallest log_prob between a set distribution and another set
# log_prob instead of huber loss as a distance metric
def prob_chamfer_distance(set_dist, set_real):
    batch_shape = tf.shape(set_real)[:-2]
    batch_total = tf.reduce_prod(batch_shape)
    element_total = tf.shape(set_real)[-2]

    # flatten our batch dimensions so we just have [batch, elements, features]
    x = tf.reshape(set_real, (-1,) + tuple(set_real.shape[-2:]))
    x_dist = tfd.BatchReshape(distribution=set_dist, batch_shape=[batch_total, element_total], validate_args=True)

    # tile the elements so we can compare the log_prob of every element to every other element
    repeated_elems = tf.transpose(tf.repeat(tf.expand_dims(x, axis=-2), element_total, axis=-2), perm=(1, 0, 2, 3))
    pointwise_log_prob = x_dist.log_prob(repeated_elems)
    log_probs = tf.transpose(pointwise_log_prob, (1, 2, 0))

    minimum_square_distance_a_to_b = tf.reduce_max(input_tensor=log_probs, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_max(input_tensor=log_probs, axis=-2)

    setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

    out_shape = tf.shape(set_real)[:-2]
    batch_shaped = tf.reshape(setwise_distance, shape=out_shape)
    return batch_shaped


if __name__ == '__main__':
    # simple set to ensure math is checking out
    simple_mean = tf.constant([[[-0.5], [1.5]], [[1.0], [0.0]]])
    simple_dist = tfd.Independent(tfd.Normal(simple_mean, 1), 1)

    closest_prob = simple_dist.log_prob(simple_mean)
    expected = (tf.reduce_mean(input_tensor=closest_prob, axis=-1) + tf.reduce_mean(input_tensor=closest_prob, axis=-1))

    # same set but with elements swapped, to make sure the minimum permutation is being found
    inverted_true = tf.constant([[[1.5], [-0.5]], [[0.0], [1.0]]])
    actual = prob_chamfer_distance(simple_dist, inverted_true)

    eq = tf.assert_equal(expected, actual)

    # test some functionality with a big set with a batch like the unit encoder
    num_units = 200
    true_set = tf.random.normal([10, 10, num_units, 65])

    mean = tf.random.normal([10, 10, num_units, 65])
    dist = tfd.Independent(tfd.Normal(mean, 1), 1)

    guy = prob_chamfer_distance(dist, true_set)

    # compare against manually comparing every point against every other point to double check transposes are doing as expected

    probs = []
    for i in range(num_units):
        dist_probs = []
        unit = true_set[:, :, i:i+1, :]
        unit = tf.repeat(unit, num_units, axis=2)
        probs.append(dist.log_prob(unit))

    all_probs = tf.stack(probs, axis=3)
    minimum_square_distance_a_to_b = tf.reduce_min(input_tensor=all_probs, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min(input_tensor=all_probs, axis=-2)

    setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

    eq = tf.assert_equal(guy, setwise_distance)

    pass
