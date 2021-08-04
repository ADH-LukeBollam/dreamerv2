from pysc2.lib import actions
import tensorflow as tf
import numpy as np
from common import dists


class Sc2RandomAgent:
    def __init__(self, action_spec):
        # use one big dist, then split it
        self._action_dists = {key: dists.OneHotDist(tf.zeros(value.shape)) for (key, value) in action_spec.spaces.items()}

        # use continuous distribution to randomise actions, because its constrained by available actions
        self._action_dists['action_id'] = dists.TruncNormalDist(tf.zeros(action_spec.spaces['action_id'].shape), 0.5, 0, 1)

    def __call__(self, obs, state=None, mode=None):
        output = {}
        for k in self._action_dists:
            output[k] = self._action_dists[k].sample(len(obs['reset']))

        # zero out unavailable actions
        invalid_masked = output['action_id'] * obs['available_actions']
        indices = tf.argmax(invalid_masked, axis=-1)
        output['action_id'] = tf.one_hot(indices, depth=tf.shape(invalid_masked)[-1])

        return output, None
