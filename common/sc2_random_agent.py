from pysc2.lib import actions
import tensorflow as tf
import numpy as np

from pysc2.lib.actions import get_args_indices_lookup


class Sc2RandomAgent:
    def __init__(self, screen_size, minimap_size):
        self.all_args = get_args_indices_lookup(screen_size, minimap_size)

    def __call__(self, obs, state=None, mode=None):
        num_steps = len(obs['reset'])
        output = {}

        actions = []
        args = []

        for i in range(num_steps):
            av_act = np.squeeze(np.argwhere(obs['available_actions'][0]))
            action_id = np.random.choice(av_act)
            actions.append(np.array([action_id], dtype=np.int))

            arg_set = np.zeros([item for sublist in list(self.all_args.values())[-1] for item in sublist][-1], dtype=np.int)
            arg_indices = [np.random.randint(arg_range[0], arg_range[1]) for ranges in self.all_args.values() for arg_range in ranges]
            # arg_indices = np.concatenate([np.random.randint(arg_range[0], arg_range[1]) for arg_range in arg_set for arg_set in self.all_args.values()], axis=-1)
            arg_set[arg_indices] = 1

            # args = [[np.random.randint(0, size) for size in arg.sizes]
            #         for arg in self.action_spec.functions[function_id].args]

            args.append(arg_set)

        output['action_id'] = np.stack(actions)
        output['action_args'] = np.stack(args)
        return output, None
