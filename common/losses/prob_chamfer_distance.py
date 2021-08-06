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

    # create a batch to pass into the distribution

    batch_axis = tf.shape(batch_shape)
    remaining_shape = tf.range(batch_axis+1, batch_axis+3)
    in_perm = tf.concat([batch_axis, batch_shape, remaining_shape], axis=0)
    batched_in_trans = tf.transpose(tf.repeat(tf.expand_dims(set_real, axis=-2), num_units, axis=-2), perm=in_perm)

    point_batch_log_prob = set_dist.log_prob(batched_in_trans)

    out_perm = tf.concat([tf.range(1, tf.rank(point_batch_log_prob)), [0]], axis=0)
    log_probs = tf.transpose(point_batch_log_prob, out_perm)

    minimum_square_distance_a_to_b = tf.reduce_min(input_tensor=log_probs, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min(input_tensor=log_probs, axis=-2)

    setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
    return setwise_distance


if __name__ == '__main__':
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
