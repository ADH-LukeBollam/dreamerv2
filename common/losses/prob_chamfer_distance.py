import tensorflow as tf
from tensorflow_probability import distributions as tfd

# this is modified from tensorflows chamfer distance func
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py


# modification of chamfer distance to calculate smallest log_prob between a set distribution and another set
# log_prob instead of huber loss as a distance metric
def prob_chamfer_distance(set_dist, set_real, sizes):
    batch_shape = tf.shape(set_real)[:-2]
    batch_total = tf.reduce_prod(batch_shape)
    element_total = tf.shape(set_real)[-2]


    # compare each element with every other element
    probs = []
    for i in range(element_total):
        unit = set_real[:, :, i:i + 1, :]
        probs.append(set_dist.log_prob(unit))
    log_probs = tf.stack(probs, axis=3)

    # flatten our batch dimensions so we just have [batch, elements, features]
    log_probs = tf.reshape(log_probs, (-1, element_total, element_total))

    # remove the padded values before finding the min distance, otherwise the model can abuse the padding to
    # achieve lower chamfer loss and not actually learn anything
    # slice off the known extras from our tensor, otherwise raggedTensor throws an error if the final ragged
    # tensor can be squeezed smaller than the initial size (ie. at least one row / column needs to be current size)
    sizes_flat = tf.reshape(sizes, (-1))
    largest_unpadded_dim = tf.reduce_max(sizes_flat)
    log_probs_trimmed = log_probs[:, :largest_unpadded_dim, :largest_unpadded_dim]

    row_sizes = tf.repeat(sizes_flat, sizes_flat)
    log_probs_ragged = tf.RaggedTensor.from_tensor(log_probs_trimmed, lengths=(sizes_flat, row_sizes))

    minimum_square_distance_a_to_b = tf.reduce_max(input_tensor=log_probs_ragged, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_max(input_tensor=log_probs_ragged, axis=-2)

    setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

    out_shape = tf.shape(set_real)[:-2]
    batch_shaped = tf.reshape(setwise_distance, shape=out_shape)
    return batch_shaped


if __name__ == '__main__':
    # simple set to ensure math is checking out
    simple_mean = tf.constant([[[-0.5], [1.5], [0]], [[1.0], [0.0], [0]]], tf.float32)
    simple_dist = tfd.Independent(tfd.Normal(simple_mean, 1), 1)

    # slice off padding
    closest_prob = simple_dist.log_prob(simple_mean)
    closest_prob_unpadded = simple_mean[:, :2]
    expected = (tf.reduce_mean(input_tensor=closest_prob, axis=-1) + tf.reduce_mean(input_tensor=closest_prob, axis=-1))

    # same set but with elements swapped, to make sure the minimum permutation is being found
    inverted_true = tf.constant([[[1.5], [-0.5], [0]], [[0.0], [1.0], [0]]], tf.float32)
    # actual = prob_chamfer_distance(simple_dist, inverted_true, [2, 2])

    # test some functionality with a big set with a batch like the unit encoder
    num_units = 200
    true_set = tf.random.normal([10, 10, num_units, 333])

    mean = tf.random.normal([10, 10, num_units, 333])
    dist = tfd.Independent(tfd.Normal(mean, 1), 1)

    sizes = tf.random.uniform([10, 10], 50, 150, dtype=tf.int32)

    actual = prob_chamfer_distance(dist, true_set, sizes)

    # compare against manually comparing every point against every other point to double check transposes are doing as expected

    probs = []
    for i in range(num_units):
        dist_probs = []
        unit = true_set[:, :, i:i+1, :]
        # unit = tf.repeat(unit, num_units, axis=2)
        probs.append(dist.log_prob(unit))

    sizes_flat = tf.reshape(sizes, (-1))

    all_probs = tf.stack(probs, axis=3)
    all_probs = tf.reshape(all_probs, (-1,) + tuple(all_probs.shape[-2:]))
    largest_unpadded_dim = tf.reduce_max(sizes_flat)
    log_probs_trimmed = all_probs[:, :largest_unpadded_dim, :largest_unpadded_dim]

    row_sizes = tf.repeat(sizes_flat, sizes_flat)
    log_probs_ragged = tf.RaggedTensor.from_tensor(log_probs_trimmed, lengths=(sizes_flat, row_sizes))

    minimum_square_distance_a_to_b = tf.reduce_max(input_tensor=log_probs_ragged, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_max(input_tensor=log_probs_ragged, axis=-2)

    setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

    out_shape = tf.shape(true_set)[:-2]
    expected = tf.reshape(setwise_distance, shape=out_shape)

    eq = tf.assert_equal(actual, expected)

    pass
