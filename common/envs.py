import os
import threading

import gym
import numpy as np

from pysc2.env import sc2_env, available_actions_printer
from pysc2.lib import actions
from pysc2.lib.buffs import get_buff_embed_lookup
from pysc2.lib.features import ScreenFeatures, Player, FeatureUnit
from pysc2.lib.units import get_unit_embed_lookup


class DMC:

    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        os.environ['MUJOCO_GL'] = 'egl'
        domain, task = name.split('_', 1)
        if domain == 'cup':  # Only domain with multiple words.
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        action = action['action']
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            if time_step.last():
                break
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


class Atari:
    LOCK = threading.Lock()

    def __init__(
        self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
        life_done=False, sticky_actions=True, all_actions=False):
        assert size[0] == size[1]
        import gym.wrappers
        import gym.envs.atari
        if name == 'james_bond':
            name = 'jamesbond'
        with self.LOCK:
            env = gym.envs.atari.AtariEnv(
                game=name, obs_type='image', frameskip=1,
                repeat_action_probability=0.25 if sticky_actions else 0.0,
                full_action_space=all_actions)
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
        mean = env.unwrapped.get_action_meanings()

        env = gym.wrappers.AtariPreprocessing(
            env, noops, action_repeat, size[0], life_done, grayscale)
        self._env = env
        self._grayscale = grayscale

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'image': self._env.observation_space,
            'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({'action': self._env.action_space})

    def close(self):
        return self._env.close()

    def reset(self):
        with self.LOCK:
            image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.env._get_ram()}
        return obs

    def step(self, action):
        action = action['action']
        image, reward, done, info = self._env.step(action)
        if self._grayscale:
            image = image[..., None]
        obs = {'image': image, 'ram': self._env.env._get_ram()}
        return obs, reward, done, info

    def render(self, mode):
        return self._env.render(mode)


class Sc2:
    def __init__(self, map_name, screen_size, minimap_size, steps_per_action, steps_per_episode, fog, visualise):
        from absl import flags
        flags.FLAGS.mark_as_parsed()
        env = sc2_env.SC2Env(
            map_name=map_name,
            battle_net_map=False,
            players=[sc2_env.Agent(sc2_env.Race.random, 'agent')],
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=screen_size,
                feature_minimap=minimap_size,
                use_feature_units=True,  # units in screen view
                use_raw_units=False  # all units including outside screen / invis
            ),
            step_mul=steps_per_action,
            game_steps_per_episode=steps_per_episode,
            disable_fog=fog,
            visualize=visualise)
        env = available_actions_printer.AvailableActionsPrinter(env)
        self._env = env
        self.unit_embed_lookup = get_unit_embed_lookup()
        self.buff_embed_lookup = get_buff_embed_lookup()

    @property
    def available_actions(self):
        return self._env.action_spec()[0]

    @property
    def observation_space(self):
        image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        return gym.spaces.Dict({'image': image})

    @property
    def action_space(self):
        action_id = gym.spaces.Box(-1, 1, (1,), dtype=np.int)
        action_args = gym.spaces.Box(-1, 1, (10,), dtype=np.int)
        action_arg_shapes = gym.spaces.Box(-1, 1, (5,), dtype=np.int)
        return gym.spaces.Dict({'action_id': action_id, 'action_args': action_args, 'action_arg_shapes': action_arg_shapes})

    def step(self, action):
        args = []
        # rebuild action args
        current_arg = 0
        for c in action['action_arg_shapes']:
            if c == 0:
                break
            a = []
            for i in range(c):
                a.append(action['action_args'][current_arg])
                current_arg += 1
            args.append(a)

        sc2_action = actions.FunctionCall(action['action_id'][0], args)

        timestep = self._env.step([sc2_action])[0]
        obs = self.collect_sc_observation(timestep)
        reward = timestep.reward

        done = False
        if timestep.last():
            done = True

        info = {}
        return obs, reward, done, info

    def reset(self):
        timestep = self._env.reset()[0]
        obs = self.collect_sc_observation(timestep)

        return obs

    def collect_sc_observation(self, timestep):
        obs = {}

        # store available actions, pad up to 30
        actions_pad_size = 30
        av_actions = timestep.observation.available_actions
        obs['available_actions_count'] = np.array([np.size(av_actions, 0)])
        action_pad = actions_pad_size - np.size(av_actions, 0)
        obs['available_actions'] = np.concatenate([av_actions, np.zeros(action_pad, dtype=np.int)])

        # screen features
        screen_feat = timestep.observation.feature_screen
        obs['screen'] = np.stack([screen_feat.visibility_map,
                                  screen_feat.creep,
                                  screen_feat.height_map,
                                  screen_feat.buildable,
                                  screen_feat.pathable],
                                 axis=2)

        # minimap features
        mini_feat = timestep.observation.feature_minimap
        obs['mini'] = np.stack([mini_feat.visibility_map,
                                mini_feat.creep,
                                mini_feat.height_map,
                                mini_feat.buildable,
                                mini_feat.pathable,
                                mini_feat.player_relative,
                                mini_feat.camera],
                               axis=2)

        # player features
        player_feat = timestep.observation.player
        obs['player'] = player_feat[[Player.minerals, Player.vespene, Player.food_used, Player.food_cap, Player.larva_count, Player.warp_gate_count]]

        # units on the screen => limit to 200
        size = 200
        units = timestep.observation.feature_units
        units = units[:, [self.unit_embed_lookup(FeatureUnit.unit_type),
                          FeatureUnit.alliance,
                          FeatureUnit.health_ratio,
                          FeatureUnit.shield_ratio,
                          FeatureUnit.energy_ratio,
                          FeatureUnit.x,
                          FeatureUnit.y,
                          FeatureUnit.radius,
                          FeatureUnit.is_selected,
                          FeatureUnit.is_blip,
                          FeatureUnit.build_progress,
                          FeatureUnit.is_powered,
                          FeatureUnit.mineral_contents,
                          FeatureUnit.vespene_contents,
                          FeatureUnit.cargo_space_taken,
                          FeatureUnit.cargo_space_max,
                          FeatureUnit.is_flying,
                          FeatureUnit.is_burrowed,
                          FeatureUnit.cloak,
                          FeatureUnit.hallucination,
                          FeatureUnit.attack_upgrade_level,
                          FeatureUnit.armor_upgrade_level,
                          FeatureUnit.shield_upgrade_level,
                          FeatureUnit.buff_id_0,
                          FeatureUnit.buff_id_1
                          ]]

        obs['unit_count'] = np.array([np.size(units, 0)])
        unit_dim = size - np.size(units, 0)
        feature_dim = np.size(units, 1)

        obs['units'] = np.concatenate([units, np.zeros((unit_dim, feature_dim))])

        return obs

    def close(self):
        return self._env.close()


class Dummy:

    def __init__(self):
        pass

    @property
    def observation_space(self):
        image = gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8)
        return gym.spaces.Dict({'image': image})

    @property
    def action_space(self):
        action = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        return gym.spaces.Dict({'action': action})

    def step(self, action):
        obs = {'image': np.zeros((64, 64, 3))}
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        obs = {'image': np.zeros((64, 64, 3))}
        return obs


class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class NormalizeAction:

    def __init__(self, env, key='action'):
        self._env = env
        self._key = key
        space = env.action_space[key]
        self._mask = np.isfinite(space.low) & np.isfinite(space.high)
        self._low = np.where(self._mask, space.low, -1)
        self._high = np.where(self._mask, space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        space = gym.spaces.Box(low, high, dtype=np.float32)
        return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

    def step(self, action):
        orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
        orig = np.where(self._mask, orig, action[self._key])
        return self._env.step({**action, self._key: orig})


class OneHotAction:

    def __init__(self, env, key='action'):
        assert isinstance(env.action_space[key], gym.spaces.Discrete)
        self._env = env
        self._key = key
        self._random = np.random.RandomState()

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space[self._key].n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        space.n = shape[0]
        return gym.spaces.Dict({**self._env.action_space.spaces, self._key: space})

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env, key='reward'):
        assert key not in env.observation_space.spaces
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        space = gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32)
        return gym.spaces.Dict({
            **self._env.observation_space.spaces, self._key: space})

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class ResetObs:

    def __init__(self, env, key='reset'):
        assert key not in env.observation_space.spaces
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        space = gym.spaces.Box(0, 1, (), dtype=np.bool)
        return gym.spaces.Dict({
            **self._env.observation_space.spaces, self._key: space})

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reset'] = np.array(False, np.bool)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reset'] = np.array(True, np.bool)
        return obs
