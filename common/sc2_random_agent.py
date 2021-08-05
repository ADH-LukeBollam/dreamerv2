from pysc2.lib import actions
import tensorflow as tf
import numpy as np
from common import dists


class Sc2RandomAgent:
    def __init__(self, action_spec):
        # It's slow to call a dist many times, so use one big one, then split it
        size = sum([value.shape[0] for value in action_spec.values()])
        self._action_spec = {key: value.shape[0] for (key, value) in action_spec.items()}
        self._dist = dists.TruncNormalDist(tf.zeros(size), 0.5, 0, 1)

    def __call__(self, obs, state=None, mode=None):
        output = {}

        rand = self._dist.sample(len(obs['reset']))
        index = 0
        for k in self._action_spec:
            size = self._action_spec[k]
            if k == 'action_id':
                # zero out unavailable actions
                invalid_masked = rand[:, index:index+size] * obs['available_actions']
                output['action_id'] = tf.one_hot(tf.argmax(invalid_masked, axis=-1), depth=tf.shape(invalid_masked)[-1])
            else:
                size = self._action_spec[k]
                output[k] = tf.one_hot(tf.argmax(rand[:, index:index+size], axis=-1), depth=size)
            index += size



        return output, None
