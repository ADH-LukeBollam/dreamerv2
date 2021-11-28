import numpy as np
import tensorflow as tf


class A2CDriver:

    def __init__(self, envs, **kwargs):
        self._envs = envs
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._actspace = envs[0].action_space
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._dones = [True] * len(self._envs)
        self._eps = [None] * len(self._envs)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0

        while step < steps or episode < episodes:
            for i, done in enumerate(self._dones):
                if done:
                    ob = self._envs[i].reset()
                    state = {**ob, 'reward': 0.0, 'discount': 1.0, 'done': False}
                    state = {k: self._convert(v) for k, v in state.items()}
                    [callback(state, **self._kwargs) for callback in self._on_resets]
                    self._obs[i] = state
                    self._eps[i] = [state]

            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
            metrics, self._obs = policy(obs, self._envs)

            for i, obs in enumerate(self._obs):
                [callback(metrics, **self._kwargs) for callback in self._on_steps]
                self._eps[i].append(obs)

                if obs['done']:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [callback(ep, **self._kwargs) for callback in self._on_episodes]

            self._dones = [obs['done'].item() for obs in self._obs]
            episode += sum(self._dones)
            step += len(self._dones)

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
