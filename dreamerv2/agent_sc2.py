import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl


class Sc2Agent(common.Module):

    def __init__(self, config, logger, actspce, step, dataset):
        self.config = config
        self._logger = logger
        self._action_space = actspce
        self._num_act = actspce.n if hasattr(actspce, 'n') else actspce.shape[0]
        self._should_expl = elements.Until(int(
            config.expl_until / config.action_repeat))
        self._counter = step
        with tf.device('cpu:0'):
            self.step = tf.Variable(int(self._counter), tf.int64)
        self._dataset = dataset
        self.wm = WorldModel(self.step, config)
        self._task_behavior = ActorCritic(config, self.step, self._num_act)
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(actspce),
            plan2explore=lambda: expl.Plan2Explore(
                config, self.wm, self._num_act, self.step, reward),
            model_loss=lambda: expl.ModelLoss(
                config, self.wm, self._num_act, self.step, reward),
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
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self.step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
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


class WorldModel(common.Module):

    def __init__(self, step, config):
        self.step = step
        self.config = config
        self.rssm = common.RSSM(**config.rssm)
        self.heads = {}
        shape = config.image_size + (1 if config.grayscale else 3,)
        self.encoder = common.Sc2Encoder(**config.encoder)
        self.heads['available_actions'] = None
        self.heads['screen'] = common.ConvDecoder(shape, **config.decoder)
        self.heads['mini'] = common.ConvDecoder(shape, **config.decoder)
        self.heads['player'] = None
        self.heads['units'] = None
        self.heads['reward'] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads['discount'] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.Optimizer('model', **config.model_opt)

    def train(self, data, state=None):
        with tf.GradientTape() as model_tape:
            model_loss, state, outputs, metrics = self.loss(data, state)
        modules = [self.encoder, self.rssm, *self.heads.values()]
        metrics.update(self.model_opt(model_tape, model_loss, modules))
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data['action'], state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else tf.stop_gradient(feat)
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

    def imagine(self, policy, start, horizon):
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = self.rssm.get_feat(state)
            action = policy(tf.stop_gradient(feat)).sample()
            succ = self.rssm.img_step(state, action)
            return succ, feat, action

        feat = 0 * self.rssm.get_feat(start)
        action = policy(feat).mode()
        succs, feats, actions = common.static_scan(
            step, tf.range(horizon), (start, feat, action))
        states = {k: tf.concat([
            start[k][None], v[:-1]], 0) for k, v in succs.items()}
        if 'discount' in self.heads:
            discount = self.heads['discount'](feats).mean()
        else:
            discount = self.config.discount * tf.ones_like(feats[..., 0])
        return feats, states, actions, discount

    @tf.function
    def preprocess(self, obs):
        dtype = prec.global_policy().compute_dtype
        obs = obs.copy()

        # screen preproc
        screen_vis = tf.one_hot(obs['screen'][:, :, :, :, 0], 3, dtype=dtype)
        screen_height = tf.cast(obs['screen'][:, :, :, :, 1:2], dtype) / 255.0 - 0.5
        obs['screen'] = tf.concat([screen_vis, screen_height, tf.cast(obs['screen'][:, :, :, :, 2:5], dtype)], axis=4)

        # minimap preproc
        mini_vis = tf.one_hot(obs['mini'][:, :, :, :, 0], 4, dtype=dtype)
        mini_height = tf.cast(obs['mini'][:, :, :, :, 1:2], dtype) / 255.0 - 0.5
        mini_relative = tf.one_hot(obs['mini'][:, :, :, :, 2], 5, dtype=dtype)
        obs['mini'] = tf.concat([mini_vis, mini_height, mini_relative, tf.cast(obs['mini'][:, :, :, :, 2:5], dtype)], axis=4)

        # unit preproc
        pp_unit_features = []
        pp_unit_features.append(obs['units'][:, :, :, 0:1])   # unit ids
        pp_unit_features.append(tf.one_hot(tf.cast(obs['units'][:, :, :, 1], tf.int32) - 1, 4, dtype=dtype))    # alliance: self = 1, ally = 2, neutral, enemy
        pp_unit_features.append(obs['units'][:, :, :, 2:5])   # health / shield / energy
        pp_unit_features.append(obs['units'][:, :, :, 5:6] / float(self.config.screen_size) - 0.5)  # x pos
        pp_unit_features.append(obs['units'][:, :, :, 6:7] / float(self.config.screen_size) - 0.5)  # y pos
        pp_unit_features.append(obs['units'][:, :, :, 7:8] / 5.0 - 0.5)  # radius: biggest units (command centers) have radius of 5
        pp_unit_features.append(obs['units'][:, :, :, 8:12])   # is_selected / is_blip / build_progress / is_powered
        pp_unit_features.append(obs['units'][:, :, :, 12:13] / 1800.0 - 0.5)     # mineral count
        pp_unit_features.append(obs['units'][:, :, :, 13:14] / 2250.0 - 0.5)     # vespene count
        pp_unit_features.append(obs['units'][:, :, :, 14:15] / 8.0 - 0.5)        # cargo taken
        pp_unit_features.append(obs['units'][:, :, :, 15:16] / 8.0 - 0.5)        # cargo max
        pp_unit_features.append(obs['units'][:, :, :, 16:19])     # is_flying / is_burrowed / is_in_cargo
        pp_unit_features.append(tf.one_hot(tf.cast(obs['units'][:, :, :, 19], tf.int32) - 1, 4, dtype=dtype))     # cloak: Cloaked = 1, CloakedDetected = 2, NotCloaked = 3, Unknown = 4, -1 so its 0 indexed
        pp_unit_features.append(obs['units'][:, :, :, 20:21])     # is_hallucination
        pp_unit_features.append(obs['units'][:, :, :, 21:22] / 3.0 - 0.5)        # attack upgrade
        pp_unit_features.append(obs['units'][:, :, :, 22:23] / 3.0 - 0.5)        # armour upgrade
        pp_unit_features.append(obs['units'][:, :, :, 23:24] / 3.0 - 0.5)        # shield upgrade
        pp_unit_features.append(obs['units'][:, :, :, 24:63])
        obs['units'] = tf.concat(pp_unit_features, axis=3)

        obs['reward'] = getattr(tf, self.config.clip_rewards)(obs['reward'])
        if 'discount' in obs:
          obs['discount'] *= self.config.discount
        return obs

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


class ActorCritic(common.Module):

    def __init__(self, config, step, num_actions):
        self.config = config
        self.step = step
        self.num_actions = num_actions
        self.actor = common.MLP(num_actions, **config.actor)
        self.critic = common.MLP([], **config.critic)
        if config.slow_target:
            self._target_critic = common.MLP([], **config.critic)
            self._updates = tf.Variable(0, tf.int64)
        else:
            self._target_critic = self.critic
        self.actor_opt = common.Optimizer('actor', **config.actor_opt)
        self.critic_opt = common.Optimizer('critic', **config.critic_opt)

    def train(self, world_model, start, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        with tf.GradientTape() as actor_tape:
            feat, state, action, disc = world_model.imagine(self.actor, start, hor)
            reward = reward_fn(feat, state, action)
            target, weight, mets1 = self.target(feat, action, reward, disc)
            actor_loss, mets2 = self.actor_loss(feat, action, target, weight)
        with tf.GradientTape() as critic_tape:
            critic_loss, mets3 = self.critic_loss(feat, action, target, weight)
        metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    def actor_loss(self, feat, action, target, weight):
        metrics = {}
        policy = self.actor(tf.stop_gradient(feat))
        if self.config.actor_grad == 'dynamics':
            objective = target
        elif self.config.actor_grad == 'reinforce':
            baseline = self.critic(feat[:-1]).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = policy.log_prob(action)[:-1] * advantage
        elif self.config.actor_grad == 'both':
            baseline = self.critic(feat[:-1]).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = policy.log_prob(action)[:-1] * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
            metrics['actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent[:-1]
        actor_loss = -(weight[:-1] * objective).mean()
        metrics['actor_ent'] = ent.mean()
        metrics['actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    def critic_loss(self, feat, action, target, weight):
        dist = self.critic(feat)[:-1]
        target = tf.stop_gradient(target)
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {'critic': dist.mode().mean()}
        return critic_loss, metrics

    def target(self, feat, action, reward, disc):
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
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
