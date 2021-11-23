import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
from models.set_prior import SetPrior
from pysc2.lib.features import Visibility, Effects, PlayerRelative
from pysc2.lib.units import get_unit_embed_lookup
from losses.prob_chamfer_distance import prob_chamfer_distance
import numpy as np


class A2CAgent(common.Module):
    def __init__(self, config, logger, actspce, step, action_required_args, env):
        self.config = config
        self.step = step

        self._logger = logger
        self._action_space = actspce
        self._act_size = sum([self._action_space[a].n for a in self._action_space])

        self.arg_keys = [key for (key, value) in actspce.items() if key != 'action_id']

        self._should_expl = elements.Until(int(
            config.expl_until / config.action_repeat))
        self._counter = step
        with tf.device('cpu:0'):
            self.step = tf.Variable(int(self._counter), tf.int64)

        self.encoder = common.Sc2Encoder(**config.encoder)

        self.type_actor = common.Sc2MLP(self._action_space['action_id'].n, **config.type_actor)
        arg_space_sizes = {key: (value.n,) for (key, value) in self._action_space.items() if key != 'action_id'}
        self.arg_actor = common.Sc2CompositeMLP(arg_space_sizes, **config.arg_actor)
        self.critic = common.Sc2MLP([], **config.critic)
        self.action_required_args = action_required_args

        # create a lookup to find which actions each arg is used for, because its a pain to do action-to-arg in TF when iterating by arg types
        self.actions_using_arg = {}
        for arg in arg_space_sizes:
            acts = []
            for act in (range(self._action_space['action_id'].n)):
                if arg in action_required_args[act]:
                    acts.append(act)
            self.actions_using_arg[arg] = tf.constant(acts, dtype=tf.int32)

        self.arg_keys = list(self.actions_using_arg.keys())

        if config.slow_target:
            self._target_critic = common.Sc2MLP([], **config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic
        self.type_actor_opt = common.Optimizer('type_actor', **config.type_actor_opt)
        self.arg_actor_opt = common.Optimizer('arg_actor', **config.arg_actor_opt)
        self.critic_opt = common.Optimizer('critic', **config.critic_opt)

        # Train step to initialize variables including optimizer statistics.
        act = {k: np.zeros(v.shape) for k, v in actspce.items()}
        for a in act:
            act[a][0] = 1

        env.reset()
        (ob, rew, done, info) = env.step(act)
        disc = info.get('discount', np.array(1 - float(done)))
        obs = {**ob, 'reward': rew, 'discount': disc, 'done': done}
        obs = {k: np.expand_dims(self._convert(v), 0) for k, v in obs.items()}
        self.train(obs, env)

    def get_sc2_action(self, type_actor, arg_actor, feat, available_actions, should_sample, mode):
        action_norm, action_onehot = type_actor(feat)

        if should_sample:
            action_prob = action_norm.sample()
        else:
            action_prob = action_norm.mode()

        noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
        noise = noise[mode]
        action_prob = tf.nn.softmax(common.sc2_action_noise(action_prob, noise))

        available_action_types = tf.cast(available_actions, tf.float32)
        masked_actions = action_prob * available_action_types + available_action_types  # scale unavailable actions to 0, then available actions get a +1 bonus so that if everything is 0, they are still higher
        chosen_action_id = tf.argmax(masked_actions, axis=-1, output_type=tf.int32)
        chosen_action_oh = tf.one_hot(chosen_action_id, depth=tf.shape(action_prob)[-1])

        feat_action = tf.concat([feat, tf.cast(chosen_action_oh, feat.dtype)], -1)
        arg_policy_out = arg_actor(tf.stop_gradient(feat_action))

        args = {}
        for a in self.arg_keys:
            if should_sample:
                arg_val = arg_policy_out[a].sample()
            else:
                arg_val = arg_policy_out[a].mode()

            arg_val = common.sc2_arg_noise(arg_val, noise)
            args[a] = arg_val

        return chosen_action_oh, action_prob, args

    @tf.function
    def policy(self, obs, mode='train'):
        tf.py_function(lambda: self.step.assign(
            int(self._counter), read_value=False), [], [])

        embed = self.encoder(self.preprocess(obs))
        action_oh, action_prob, args = self.get_sc2_action(self.type_actor, self.arg_actor, embed, obs['available_actions'], False, mode)

        return embed, action_prob, action_oh, args

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()

        # screen preproc
        # obs['screen_visibility'] = tf.one_hot(obs['screen'][..., 0], len(Visibility), dtype=dtype)  # screen visibility
        # obs['screen_height'] = tf.cast(obs['screen'][..., 1:2], dtype=dtype) / 255.0 - 0.5  # screen height
        # obs['screen_creep'] = tf.cast(obs['screen'][..., 2:3], dtype=dtype)  # creep / buildable / pathable
        # obs['screen_buildable'] = tf.cast(obs['screen'][..., 3:4], dtype=dtype)
        # obs['screen_pathable'] = tf.cast(obs['screen'][..., 4:5], dtype=dtype)
        # obs['screen_effects'] = tf.one_hot(obs['screen'][..., 5], len(Effects), dtype=dtype)  # screen effects one-hot
        del obs['screen']
        del obs['mini']
        del obs['player']

        # # minimap preproc
        # pp_mini_feat = []
        # pp_mini_feat.append(tf.one_hot(obs['mini'][..., 0], len(Visibility), dtype=tf.float32))  # minimap visibility
        # pp_mini_feat.append(tf.cast(obs['mini'][..., 1:2], dtype=tf.float32) / 255.0 - 0.5)  # minimap height
        # pp_mini_feat.append(tf.one_hot(obs['mini'][..., 2], len(PlayerRelative), dtype=tf.float32))  # minimap player relative unit alliance
        # pp_mini_feat.append(tf.cast(obs['mini'][..., 3:7], dtype=tf.float32) * 1.0)  # creep / buildable / pathable / camera
        # obs['mini'] = tf.cast(tf.concat(pp_mini_feat, axis=-1), dtype=dtype)

        # unit preproc
        obs['unit_id'] = tf.one_hot(obs['units'][..., 0], len(set(get_unit_embed_lookup().values())), dtype=tf.int32)  # unit ids
        # obs['unit_alliance'] = tf.one_hot(obs['units'][..., 1] - 1, 4, dtype=dtype)  # alliance: self = 1, ally = 2, neutral, enemy, -1 so its 0 indexed
        # obs['unit_cloaked'] = tf.one_hot(obs['units'][..., 19] - 1, 4, dtype=dtype)  # cloak: Cloaked = 1, CloakedDetected = 2, NotCloaked = 3, Unknown = 4, -1 so its 0 indexed
        obs['unit_continuous'] = tf.concat([
            # tf.cast(obs['units'][..., 2:5], dtype=dtype) / 255.0 - 0.5,  # health / shield / energy are all in scale 0-255
            tf.cast(obs['units'][..., 5:6], dtype=dtype) / float(self.config.screen_size) - 0.5,  # x pos
            tf.cast(obs['units'][..., 6:7], dtype=dtype) / float(self.config.screen_size) - 0.5,  # y pos
            # tf.cast(obs['units'][..., 7:8], dtype=dtype) / 5.0 - 0.5,  # radius: biggest units (command centers) have radius of 5
            # tf.cast(obs['units'][..., 10:11], dtype=dtype),  # build_progress
            # tf.cast(obs['units'][..., 12:13], dtype=dtype) / 1800.0 - 0.5,  # mineral count
            # tf.cast(obs['units'][..., 13:14], dtype=dtype) / 2250.0 - 0.5,  # vespene count
            # tf.cast(obs['units'][..., 14:15], dtype=dtype) / 8.0 - 0.5,  # cargo taken
            # tf.cast(obs['units'][..., 15:16], dtype=dtype) / 8.0 - 0.5,  # cargo max
            # tf.cast(obs['units'][..., 21:22], dtype=dtype) / 3.0 - 0.5,  # attack upgrade
            # tf.cast(obs['units'][..., 22:23], dtype=dtype) / 3.0 - 0.5,  # armour upgrade
            # tf.cast(obs['units'][..., 23:24], dtype=dtype) / 3.0 - 0.5  # shield upgrade
        ], axis=-1)
        obs['unit_binary'] = tf.concat([
            tf.cast(obs['units'][..., 8:9], dtype=dtype),  # is_selected
            # tf.cast(obs['units'][..., 9:10], dtype=dtype),  # is_blip
            # tf.cast(obs['units'][..., 11:12], dtype=dtype),  # is_powered
            # tf.cast(obs['units'][..., 16:19], dtype=dtype),  # is_flying / is_burrowed / is_in_cargo
            # tf.cast(obs['units'][..., 20:21], dtype=dtype),  # is_hallucination
            # tf.cast(obs['units'][..., 24:65], dtype=dtype)  # boolean buffs list
        ], axis=-1)
        del obs['units']

        # player preproc
        # pp_player_features = []
        # obs['player'] = tf.cast(obs['player'], dtype=tf.float32)
        # pp_player_features.append(tf.sqrt(obs['player'][..., 0:1]) / 50)  # player minerals
        # pp_player_features.append(tf.sqrt(obs['player'][..., 1:2]) / 50)  # player gas
        # pp_player_features.append(obs['player'][..., 2:3] / 200)  # supply used
        # pp_player_features.append(obs['player'][..., 3:4] / 200)  # supply max
        # pp_player_features.append(obs['player'][..., 4:5] / 10)  # warp gates
        # pp_player_features.append(obs['player'][..., 5:6] / 20)  # larva count
        # obs['player'] = tf.cast(tf.concat(pp_player_features, axis=-1), dtype=dtype)

        obs['available_actions'] = tf.cast(obs['available_actions'], dtype=dtype)

        obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
        if 'discount' in obs:
            obs['discount'] *= self.config.discount
        return obs

    @tf.function
    def action_preprocess(self, action, args):
        return tf.concat([action, args['arg_0_0'], args['arg_0_1'], args['arg_1_0'], args['arg_1_1'],
                          args['arg_2_0'], args['arg_2_1'], args['arg_3_0'], args['arg_4_0'],
                          args['arg_5_0'], args['arg_6_0'], args['arg_7_0'], args['arg_9_0'], args['arg_10_0']], axis=-1)

    @tf.function
    def eval(self, obs, env):
        metrics = {}

        obs_embed, action_prob, action_oh, args = self.policy(obs, mode='eval')
        args['action_id'] = action_oh  # combine action and args together

        ob, reward, done, info = env.step({k: np.squeeze(v) for k, v in args.items()})

        disc = info.get('discount', np.array(1 - float(done)))
        obs = {**ob, 'reward': reward, 'discount': disc, 'done': done}
        obs = {k: np.expand_dims(self._convert(v), 0) for k, v in obs.items()}

        return metrics, obs

    @tf.function
    def train(self, obs, envs):
        metrics = {}
        with tf.GradientTape(persistent=True) as actor_tape:

            obs_embed, action_prob, action_oh, args = self.policy(obs)
            args['action_id'] = action_oh   # combine action and args together

            actions = [{k: np.array(args[k][i]) for k in args} for i in range(len(envs))]
            results = [e.step(a) for e, a in zip(envs, actions)]

            new_obs = []
            for i, (ob, rew, done, info) in enumerate(results):
                disc = info.get('discount', np.array(1 - float(done)))
                tran = {**ob, 'reward': rew, 'discount': disc, 'done': done}
                new_obs.append({k: np.expand_dims(self._convert(v), 0) for k, v in tran.items()})
            new_obs = {k: np.stack([o[k] for o in new_obs]) for k in new_obs[0]}

            target, mets1 = self.target(obs_embed, action_oh, args, obs['reward'], obs['discount'])

            type_actor_loss, mets2 = self.action_loss(obs_embed, action_oh, target)
            arg_actor_loss, mets3 = self.arg_loss(obs_embed, action_oh, args, target)

        with tf.GradientTape() as critic_tape:
            critic_loss, mets4 = self.critic_loss(obs_embed, action_oh, args, target)

        metrics.update(self.type_actor_opt(actor_tape, type_actor_loss, self.type_actor))
        metrics.update(self.arg_actor_opt(actor_tape, arg_actor_loss, self.arg_actor))
        del actor_tape
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics, new_obs

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value

    @tf.function
    def action_loss(self, feat, action, target):
        metrics = {}

        norm_policy, onehot_policy = self.type_actor(tf.stop_gradient(feat))

        baseline = self.critic(feat).mode()
        advantage = tf.stop_gradient(target - baseline)
        objective = onehot_policy.log_prob(action) * advantage

        ent = onehot_policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent
        actor_loss = -objective.mean()
        metrics[f'type_actor_ent'] = ent.mean()
        metrics[f'type_actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    @tf.function
    def arg_loss(self, feat, action, action_args, target):
        metrics = {}
        chosen_action_id = tf.argmax(action, axis=-1, output_type=tf.int32)

        feat_action = tf.concat([feat, tf.cast(action, feat.dtype)], -1)
        policy = self.arg_actor(tf.stop_gradient(feat_action))

        arg_head_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        num_args_used = 0

        for arg_key in self.arg_keys:
            baseline = self.critic(feat).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = policy[arg_key].log_prob(action_args[arg_key]) * advantage

            ent = policy[arg_key].entropy()
            ent_scale = common.schedule(self.config.actor_ent, self.step)
            objective += ent_scale * ent

            # # tile our chosen actions and check if this arg is required for each action
            actions_needing_arg = self.actions_using_arg[arg_key]
            actions_tiled = tf.tile(tf.expand_dims(chosen_action_id, -1), [1, tf.size(actions_needing_arg)])
            needed = tf.where(tf.greater(tf.math.count_nonzero(tf.equal(actions_tiled, actions_needing_arg), axis=-1), 0))

            if tf.size(needed) > 0:
                # only calculate losses for used args
                objective = tf.gather_nd(objective, needed)

                actor_loss = -tf.reduce_mean(objective, axis=0, keepdims=True)
                arg_head_losses = arg_head_losses.write(num_args_used, actor_loss)
                num_args_used += 1

            metrics[f'{arg_key}_actor_ent'] = ent.mean()
            metrics[f'{arg_key}_actor_ent_scale'] = ent_scale

        if arg_head_losses.size() > 0:
            arg_losses = arg_head_losses.concat()
            arg_loss = tf.reduce_mean(arg_losses)
        else:
            arg_loss = 0.0

        return arg_loss, metrics

    def critic_loss(self, feat, action, action_args, target):
        dist = self.critic(feat)
        target = tf.stop_gradient(target)
        critic_loss = -dist.log_prob(target).mean()
        metrics = {f'critic': dist.mode().mean()}
        return critic_loss, metrics

    def target(self, feat, action, action_args, reward, disc):
        reward = tf.cast(reward, tf.float32)
        disc = tf.cast(disc, tf.float32)
        value = self._target_critic(feat).mode()
        target = common.lambda_return(
            reward, value, disc,
            bootstrap=None, lambda_=self.config.discount_lambda, axis=0)
        metrics = {}
        metrics['reward_mean'] = reward.mean()
        metrics['reward_std'] = reward.std()
        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
