import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
from models.set_prior import SetPrior
from pysc2.lib.features import Visibility, Effects, PlayerRelative
from losses.prob_chamfer_distance import prob_chamfer_distance


class Sc2Agent(common.Module):
    def __init__(self, config, logger, actspce, step, dataset):
        self.config = config
        self._logger = logger
        self._action_space = actspce
        self._should_expl = elements.Until(int(
            config.expl_until / config.action_repeat))
        self._counter = step
        with tf.device('cpu:0'):
            self.step = tf.Variable(int(self._counter), tf.int64)
        self._dataset = dataset
        self.wm = Sc2WorldModel(self.step, config, actspce)
        self._task_behavior = Sc2ActorCritic(config, self.step, self._action_space)
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(actspce),
            plan2explore=lambda: expl.Plan2Explore(config, self.wm, self._num_act, self.step, reward),
            model_loss=lambda: expl.ModelLoss(config, self.wm, self._num_act, self.step, reward),
        )[config.expl_behavior]()
        # Train step to initialize variables including optimizer statistics.
        self.train(next(self._dataset))

    @tf.function
    def policy(self, obs, state=None, mode='train'):
        tf.py_function(lambda: self.step.assign(
            int(self._counter), read_value=False), [], [])
        if state is None:
            latent = self.wm.rssm.initial(len(obs['image']))
            action = tf.zeros((len(obs['image']), self._num_act))
            state = latent, action
        elif obs['reset'].any():
            state = tf.nest.map_structure(lambda x: x * common.pad_dims(
                1.0 - tf.cast(obs['reset'], x.dtype), len(x.shape)), state)
        latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        sample = (mode == 'train') or not self.config.eval_state_mean
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, sample)
        feat = self.wm.rssm.get_feat(latent)
        if mode == 'eval':
            actor = self._task_behavior.type_actor(feat)
            action = actor.mode()
        elif self._should_expl(self.step):
            actor = self._expl_behavior.type_actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.type_actor(feat)
            action = actor.sample()
        noise = {'train': self.config.expl_noise, 'eval': self.config.eval_noise}
        action = common.action_noise(action, noise[mode], self._action_space)
        outputs = {'action': action}
        state = (latent, action)
        return outputs, state

    @tf.function
    def train(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs['post']
        if self.config.pred_discount:  # Last step could be terminal.
            start = tf.nest.map_structure(lambda x: x[:, :-1], start)
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
        metrics.update(self._task_behavior.train(self.wm, start, reward))
        if self.config.expl_behavior != 'greedy':
            if self.config.pred_discount:
                data = tf.nest.map_structure(lambda x: x[:, :-1], data)
                outputs = tf.nest.map_structure(lambda x: x[:, :-1], outputs)
            mets = self._expl_behavior.train(start, outputs, data)[-1]
            metrics.update({'expl_' + key: value for key, value in mets.items()})
        return state, metrics

    @tf.function
    def report(self, data):
        return {'openl': self.wm.video_pred(data)}


class Sc2WorldModel(common.Module):

    def __init__(self, step, config, actspace):
        self.step = step
        self.config = config
        self.rssm = common.Sc2RSSM(**config.rssm)
        self.heads = {}
        self.act_space = actspace
        self.action_input_order = self.get_action_input_order()
        self.encoder = common.Sc2Encoder(**config.encoder)
        self.heads['available_actions'] = common.MLP(actspace['action_id'].n, **config.avl_action_head)
        self.heads['screen'] = common.ConvDecoder((config.screen_size, config.screen_size, 20), **config.decoder)
        self.heads['mini'] = common.ConvDecoder((config.mini_size, config.mini_size, 13), **config.decoder)
        self.heads['player'] = common.MLP(15, **config.player_head)
        self.heads['unit_init_set'] = SetPrior(200)     # units are padded to 200
        self.heads['units'] = None
        self.heads['reward'] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads['discount'] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.Optimizer('model', **config.model_opt)

    def get_action_input_order(self):
        action_order = ['action_id']
        args = {k: v for (k, v) in self.act_space.items() if 'arg' in k}
        args = {k: v for k, v in sorted(args.items(), key=lambda x: float(x[0].split('_')[1]) + float(x[0].split('_')[2])*0.1)}   # sort by arg type, then arg axis
        action_order += [a for a in args.keys()]
        return action_order

    def train(self, data, state=None):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        action_input = self.action_preprocess(data)
        post, prior = self.rssm.observe(embed, action_input, state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else tf.stop_gradient(feat)
            if name == 'units':
                unpadded_units = tf.where(tf.not_equal(data[name][:, :, :, 0]), 0)   # find indices where unit type not 0
                unpadded_count = tf.reduce_sum(unpadded_units, axis=-1)
                like = prob_chamfer_distance(head(inp), data[name], unpadded_count)
                likes[name] = like
                losses[name] = -like.mean()
            else:
                like = tf.cast(head(inp).log_prob(data[name]), tf.float32)
                likes[name] = like
                losses[name] = -like.mean()
        model_loss = sum(
            self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = dict(
            embed=embed, feat=feat, post=post,
            prior=prior, likes=likes, kl=kl_value)
        metrics = {f'{name}_loss': value for name, value in losses.items()}
        metrics['model_kl'] = kl_value.mean()
        metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
        metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
        return model_loss, post, outs, metrics

    def imagine(self, action_policy, arg_policy, start, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = self.rssm.get_feat(state)
            action_prob = action_policy(tf.stop_gradient(feat)).sample()

            # filter out unavailable actions using learned
            available_action_types = tf.stop_gradient(self.heads['available_actions'](feat).mode())  # dont train the world model while imagining
            masked_actions = action_prob * available_action_types
            chosen_action = tf.one_hot(tf.argmax(masked_actions, axis=-1), depth=tf.shape(action)[-1])

            feat_action = tf.concat([feat, chosen_action], -1)
            action_args = arg_policy(tf.stop_gradient(feat_action)).sample()
            succ = self.rssm.img_step(state, action, action_args)
            return succ, feat, chosen_action, action_args

        feat = 0 * self.rssm.get_feat(start)
        action = action_policy(feat).mode()
        feat_action = tf.concat([feat, action], -1)
        arg = arg_policy(feat_action).mode()
        succs, feats, actions, args = common.static_scan(
            step, tf.range(horizon), (start, feat, action, arg))
        states = {k: tf.concat([
            start[k][None], v[:-1]], 0) for k, v in succs.items()}
        if 'discount' in self.heads:
            discount = self.heads['discount'](feats).mean()
        else:
            discount = self.config.discount * tf.ones_like(feats[..., 0])
        return feats, states, actions, args, discount

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()

        # screen preproc
        pp_screen_feat = []
        pp_screen_feat.append(tf.one_hot(obs['screen'][:, :, :, :, 0], len(Visibility), dtype=tf.float32))  # screen visibility
        pp_screen_feat.append(tf.cast(obs['screen'][:, :, :, :, 1:2], dtype=tf.float32) / 255.0 - 0.5)      # screen height
        pp_screen_feat.append(tf.cast(obs['screen'][:, :, :, :, 2:5], dtype=tf.float32))                    # creep / buildable / pathable
        pp_screen_feat.append(tf.one_hot(obs['screen'][:, :, :, :, 5], len(Effects), dtype=tf.float32))     # screen effects one-hot
        obs['screen'] = tf.cast(tf.concat(pp_screen_feat, axis=4), dtype=dtype)

        # minimap preproc
        pp_mini_feat = []
        pp_mini_feat.append(tf.one_hot(obs['mini'][:, :, :, :, 0], len(Visibility), dtype=tf.float32))      # minimap visibility
        pp_mini_feat.append(tf.cast(obs['mini'][:, :, :, :, 1:2], dtype=tf.float32) / 255.0 - 0.5)          # minimap height
        pp_mini_feat.append(tf.one_hot(obs['mini'][:, :, :, :, 2], len(PlayerRelative), dtype=tf.float32))  # minimap player relative unit alliance
        pp_mini_feat.append(tf.cast(obs['mini'][:, :, :, :, 3:7], dtype=tf.float32) * 1.0)                  # creep / buildable / pathable / camera
        obs['mini'] = tf.cast(tf.concat(pp_mini_feat, axis=4), dtype=dtype)

        # unit preproc
        pp_unit_features = []
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 0:1], dtype=tf.float32))   # unit ids
        pp_unit_features.append(tf.one_hot(obs['units'][:, :, :, 1] - 1, 4, dtype=tf.float32))              # alliance: self = 1, ally = 2, neutral, enemy, -1 so its 0 indexed
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 2:5], dtype=tf.float32) / 255.0)              # health / shield / energy are all in scale 0-255
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 5:6], dtype=tf.float32) / float(self.config.screen_size) - 0.5)   # x pos
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 6:7], dtype=tf.float32) / float(self.config.screen_size) - 0.5)   # y pos
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 7:8], dtype=tf.float32) / 5.0 - 0.5)          # radius: biggest units (command centers) have radius of 5
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 8:12], dtype=tf.float32))                     # is_selected / is_blip / build_progress / is_powered
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 12:13], dtype=tf.float32) / 1800.0 - 0.5)     # mineral count
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 13:14], dtype=tf.float32) / 2250.0 - 0.5)     # vespene count
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 14:15], dtype=tf.float32) / 8.0 - 0.5)        # cargo taken
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 15:16], dtype=tf.float32) / 8.0 - 0.5)        # cargo max
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 16:19], dtype=tf.float32))                    # is_flying / is_burrowed / is_in_cargo
        pp_unit_features.append(tf.one_hot(obs['units'][:, :, :, 19] - 1, 4, dtype=tf.float32))             # cloak: Cloaked = 1, CloakedDetected = 2, NotCloaked = 3, Unknown = 4, -1 so its 0 indexed
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 20:21], dtype=tf.float32))                    # is_hallucination
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 21:22], dtype=tf.float32) / 3.0 - 0.5)        # attack upgrade
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 22:23], dtype=tf.float32) / 3.0 - 0.5)        # armour upgrade
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 23:24], dtype=tf.float32) / 3.0 - 0.5)        # shield upgrade
        pp_unit_features.append(tf.cast(obs['units'][:, :, :, 24:65], dtype=tf.float32))                    # boolean buffs list
        obs['units'] = tf.cast(tf.concat(pp_unit_features, axis=3), dtype=dtype)

        # player preproc
        pp_player_features = []
        obs['player'] = tf.cast(obs['player'], dtype=tf.float32)
        pp_player_features.append(tf.sqrt(obs['player'][:, :, 0:1]) / 50)    # player minerals
        pp_player_features.append(tf.sqrt(obs['player'][:, :, 1:2]) / 50)    # player gas
        pp_player_features.append(obs['player'][:, :, 2:3] / 200)            # supply used
        pp_player_features.append(obs['player'][:, :, 3:4] / 200)            # supply max
        pp_player_features.append(obs['player'][:, :, 4:5] / 10)             # warp gates
        pp_player_features.append(obs['player'][:, :, 5:6] / 20)             # larva count
        obs['player'] = tf.cast(tf.concat(pp_player_features, axis=2), dtype=dtype)

        obs['available_actions'] = tf.cast(obs['available_actions'], dtype=dtype)

        obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
        if 'discount' in obs:
          obs['discount'] *= self.config.discount
        return obs

    @tf.function
    def action_preprocess(self, data):
        action_input = []
        for act in self.action_input_order:
            action_input.append(data[act])
        return tf.concat(action_input, axis=-1)

    @tf.function
    def video_pred(self, data):
        data = self.preprocess(data)
        truth = data['image'][:6] + 0.5
        embed = self.encoder(data)
        states, _ = self.rssm.observe(embed[:6, :5], data['action'][:6, :5])
        recon = self.heads['observation'](
            self.rssm.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(data['action'][:6, 5:], init)
        openl = self.heads['image'](self.rssm.get_feat(prior)).mode()
        model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = tf.concat([truth, model, error], 2)
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class Sc2ActorCritic(common.Module):

    def __init__(self, config, step, act_space):
        self.config = config
        self.step = step
        self.type_actor = common.Sc2MLP(act_space['action_id'].n, **config.type_actor)
        self.type_critic = common.Sc2MLP([], **config.type_critic)
        arg_space_sizes = {key: value.n for (key, value) in act_space.items() if key != 'action_id'}
        self.arg_actor = common.Sc2CompositeMLP(arg_space_sizes, **config.arg_actor)
        self.arg_critic = common.Sc2MLP([], **config.arg_critic)
        if config.slow_target:
            self._target_critic = common.Sc2MLP([], **config.type_critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.type_critic
        self.type_actor_opt = common.Optimizer('type_actor', **config.type_actor_opt)
        self.type_critic_opt = common.Optimizer('type_critic', **config.type_critic_opt)
        self.arg_actor_opt = common.Optimizer('arg_actor', **config.arg_actor_opt)
        self.arg_critic_opt = common.Optimizer('arg_critic', **config.arg_critic_opt)

    def train(self, world_model, start, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        with tf.GradientTape() as actor_tape:
            feat, state, action, action_args, disc = world_model.imagine(self.type_actor, self.arg_actor, start, hor)
            reward = reward_fn(feat, state, action, action_args)
            target, weight, mets1 = self.target(feat, action, action_args, reward, disc)

            type_actor_loss, mets2 = self.actor_loss('type', self.type_actor, self.type_critic, feat, action, target, weight)
            feat_action = tf.concat([feat, action], -1)
            arg_actor_loss, mets3 = self.actor_loss('arg', self.arg_actor, self.arg_critic, feat_action, action_args, target, weight)

        with tf.GradientTape() as critic_tape:
            type_critic_loss, mets4 = self.critic_loss('type', self.type_critic, feat, action, action_args, target, weight)
            arg_critic_loss, mets5 = self.critic_loss('arg', self.arg_critic, feat, action, action_args, target, weight)

        metrics.update(self.type_actor_opt(actor_tape, type_actor_loss, self.type_actor))
        metrics.update(self.type_critic_opt(critic_tape, type_critic_loss, self.type_critic))
        metrics.update(self.arg_actor_opt(actor_tape, arg_actor_loss, self.type_actor))
        metrics.update(self.arg_critic_opt(critic_tape, arg_critic_loss, self.type_critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4, **mets5)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, name, actor_head, critic_head, feat, action, target, weight):
        metrics = {}
        policy = actor_head(tf.stop_gradient(feat))
        if self.config.actor_grad == 'dynamics':
            objective = target
        elif self.config.actor_grad == 'reinforce':
            baseline = critic_head(feat[:-1]).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = policy.log_prob(action)[:-1] * advantage
        elif self.config.actor_grad == 'both':
            baseline = critic_head(feat[:-1]).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = policy.log_prob(action)[:-1] * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
            metrics[f'{name}_actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent[:-1]
        actor_loss = -(weight[:-1] * objective).mean()
        metrics[f'{name}_actor_ent'] = ent.mean()
        metrics[f'{name}_actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, name, critic_head, feat, action, action_args, target, weight):
        dist = critic_head(feat)[:-1]
        target = tf.stop_gradient(target)
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {f'{name}_critic': dist.mode().mean()}
        return critic_loss, metrics

    def target(self, feat, action, action_args, reward, disc):
        reward = tf.cast(reward, tf.float32)
        disc = tf.cast(disc, tf.float32)
        value = self._target_critic(feat).mode()
        target = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1], lambda_=self.config.discount_lambda, axis=0)
        weight = tf.stop_gradient(tf.math.cumprod(tf.concat(
            [tf.ones_like(disc[:1]), disc[:-1]], 0), 0))
        metrics = {}
        metrics['reward_mean'] = reward.mean()
        metrics['reward_std'] = reward.std()
        metrics['critic_slow'] = value.mean()
        metrics['critic_target'] = target.mean()
        return target, weight, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.type_critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
