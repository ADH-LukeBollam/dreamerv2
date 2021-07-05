from pysc2.lib import actions
import tensorflow as tf
import numpy as np


class Sc2RandomAgent:
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def __call__(self, obs, state=None, mode=None):
        num_steps = len(obs['reset'])
        output = {}

        function_ids = []
        arg_shapes = []
        arg_flattened = []

        for i in range(num_steps):
            function_id = np.random.choice(obs['available_actions'][0])
            function_ids.append(np.array([function_id], dtype=np.int))

            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]

            # up to 5 required args for any action (action dependant, sometimes less are needed
            arg_shape = [len(a) for a in args]
            arg_shape_padded = np.concatenate([arg_shape, np.zeros([5 - np.size(args)], dtype=np.int)])
            arg_shapes.append(arg_shape_padded)

            # up to 10 required total args components
            flat_args = [item for sublist in args for item in sublist]
            arg_pad = 10 - len(flat_args)
            args_padded = np.concatenate([flat_args, np.zeros([arg_pad], dtype=np.int)])
            arg_flattened.append(args_padded)

        output['action_id'] = np.stack(function_ids)
        output['action_args'] = np.stack(arg_flattened)
        output['action_arg_shapes'] = np.stack(arg_shapes)
        return output, None
