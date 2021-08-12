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
        unit = set_real[:, i:i + 1, :]
        probs.append(set_dist.log_prob(unit))
    log_probs = tf.stack(probs, axis=-2)

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
    log_probs = tf.RaggedTensor.from_tensor(log_probs_trimmed, lengths=(sizes_flat, row_sizes))

    minimum_square_distance_a_to_b = tf.reduce_max(input_tensor=log_probs, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_max(input_tensor=log_probs, axis=-2)

    setwise_distance = (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
                        tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))

    out_shape = tf.shape(set_real)[:-2]
    batch_shaped = tf.reshape(setwise_distance, shape=out_shape)
    return batch_shaped


if __name__ == '__main__':
    # simple set to ensure math is checking out
    mean = tf.constant([[[0.5, 0.75], [0.1, 0.25], [0.35, 0.9]], [[0.4, 0.45], [0.5, 0.7], [0.8, 0.25]]], tf.float32)
    dist = tfd.Independent(tfd.Normal(mean, 1), 1)

    # slice off padding
    closest_prob = dist.log_prob(mean)
    expected = (tf.reduce_mean(input_tensor=closest_prob, axis=-1) + tf.reduce_mean(input_tensor=closest_prob, axis=-1))

    # same set but with elements swapped, to make sure the minimum permutation is being found
    inverted_mean = tf.constant([[[0.35, 0.9], [0.5, 0.75], [0.1, 0.25]], [[0.8, 0.25], [0.4, 0.45], [0.5, 0.7]]], tf.float32)

    actual = prob_chamfer_distance(dist, inverted_mean, [3, 3])

    eq = tf.assert_equal(actual, expected)

    # check a set that should have imbalanced distance between a=>b, b=>a
    imb_mean = tf.constant([[[0.5, 0.75], [0.1, 0.25]]], tf.float32)
    imb_dist = tfd.Independent(tfd.Normal(imb_mean, 1), 1)

    true = tf.constant([[[0.5, 0.75], [0.5, 0.75]]], tf.float32)
    closest_prob = imb_dist.log_prob(true)

    # this set has 3 matching points, with one outlier
    expected = closest_prob[0][0] + tf.reduce_mean([closest_prob[0][0], closest_prob[0][1]])

    actual = prob_chamfer_distance(imb_dist, true, [2])

    eq = tf.assert_equal(actual, expected)

    pass
