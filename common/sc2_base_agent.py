from pysc2.lib import actions
import tensorflow as tf

class Sc2BaseAgent:
  def __init__(self):
      pass

  def __call__(self, obs, state=None, mode=None):
    output = {'action': tf.zeros()}
    output['action_args']: tf.zeros([len(obs['reset']), 5])
    return output, None
