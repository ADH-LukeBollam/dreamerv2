import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import elements
import common
import expl
from models.set_prior import SetPrior
from pysc2.lib.features import Visibility, Effects, PlayerRelative
from pysc2.lib.units import get_unit_embed_lookup
from losses.prob_chamfer_distance import prob_chamfer_distance


class Sc2Agent(common.Module):
    def __init__(self, config, logger, actspce, step, dataset, action_required_args):
        self.config = config
        self._logger = logger
        self._action_space = actspce
        self._act_size = sum([self._action_space[a].n for a in self._action_space])

        self.arg_keys = [key for (key, value) in actspce.items() if key != 'action_id']

        self._should_expl = elements.Until(int(
            config.expl_until / config.action_repeat))
        self._counter = step
        with tf.device('cpu:0'):
            self.step = tf.Variable(int(self._counter), tf.int64)
        self._dataset = dataset
        self.wm = Sc2WorldModel(self.step, config, actspce, action_required_args)
        self._task_behavior = Sc2ActorCritic(config, self.step, self._action_space, action_required_args)
        reward = lambda f, s, a: self.wm.heads['reward'](f).mode()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(actspce),
            plan2explore=lambda: expl.Plan2Explore(config, self.wm, self._num_act, self.step, reward),
            model_loss=lambda: expl.ModelLoss(config, self.wm, self._num_act, self.step, reward),
        )[config.expl_behavior]()
        # Train step to initialize variables including optimizer statistics.
        self.train(next(self._dataset))

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

        return chosen_action_oh, args

    @tf.function
    def policy(self, obs, state=None, mode='train'):
        tf.py_function(lambda: self.step.assign(
            int(self._counter), read_value=False), [], [])
        if state is None:
            latent = self.wm.rssm.initial(len(obs['screen']))
            action = tf.zeros((len(obs['screen']), self._act_size))
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
            action, args = self.get_sc2_action(self._task_behavior.type_actor, self._task_behavior.arg_actor, feat, obs['available_actions'], False, mode)
        elif self._should_expl(self.step):
            action, args = self.get_sc2_action(self._expl_behavior.type_actor, self._expl_behavior.arg_actor, feat, obs['available_actions'], True, mode)
        else:
            action, args = self.get_sc2_action(self._task_behavior.type_actor, self._task_behavior.arg_actor, feat, obs['available_actions'], True, mode)

        action_vec = self.wm.action_preprocess(action, args)

        args['action_id'] = action

        state = (latent, action_vec)
        return args, state

    @tf.function
    def train(self, data, state=None):
        metrics = {}
        state, outputs, mets = self.wm.train(data, state)
        metrics.update(mets)
        start = outputs['post']
        if self.config.pred_discount:  # Last step could be terminal.
            start = tf.nest.map_structure(lambda x: x[:, :-1], start)
        reward = lambda f: self.wm.heads['reward'](f).mode()
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

    def __init__(self, step, config, actspace, act_required_args):
        self.step = step
        self.config = config
        self.rssm = common.Sc2RSSM(**config.rssm)
        self.heads = {}
        self.act_space = actspace

        self.arg_keys = [key for (key, value) in actspace.items() if key != 'action_id']

        self.num_unit_types = len(get_unit_embed_lookup())
        self.action_input_order = self.get_action_input_order()
        self.encoder = common.Sc2Encoder(**config.encoder)
        self.heads['available_actions'] = common.MLP(actspace['action_id'].n, **config.avl_action_head)
        self.heads['screen'] = common.Sc2ScreenDecoder(config.screen_size, **config.decoder)
        self.heads['mini'] = common.ConvDecoder((config.mini_size, config.mini_size, 13), **config.decoder)
        self.heads['player'] = common.MLP(6, **config.player_head)
        self.heads['unit_init_set'] = SetPrior(86)  # after embedding unit type to 16 features, units have 86 features each
        self.heads['units'] = common.UnitDecoder(**config.unit_decoder)  #
        self.heads['reward'] = common.MLP([], **config.reward_head)
        if config.pred_discount:
            self.heads['discount'] = common.MLP([], **config.discount_head)
        for name in config.grad_heads:
            assert name in self.heads, name
        self.model_opt = common.Optimizer('model', **config.model_opt)

    def get_action_input_order(self):
        action_order = ['action_id']
        args = {k: v for (k, v) in self.act_space.items() if 'arg' in k}
        args = {k: v for k, v in sorted(args.items(), key=lambda x: float(x[0].split('_')[1]) + float(x[0].split('_')[2]) * 0.1)}  # sort by arg type, then arg axis
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
        args = {'arg_0_0': data['arg_0_0'], 'arg_0_1': data['arg_0_1'], 'arg_1_0': data['arg_1_0'], 'arg_1_1': data['arg_1_1'],
                'arg_2_0': data['arg_2_0'], 'arg_2_1': data['arg_2_1'], 'arg_3_0': data['arg_3_0'], 'arg_4_0': data['arg_4_0'],
                'arg_5_0': data['arg_5_0'], 'arg_6_0': data['arg_6_0'], 'arg_7_0': data['arg_7_0'], 'arg_9_0': data['arg_9_0'], 'arg_10_0': data['arg_10_0']}

        action_input = self.action_preprocess(data['action_id'], args)
        post, prior = self.rssm.observe(embed, action_input, state)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        assert len(kl_loss.shape) == 0
        likes = {}
        losses = {'kl': kl_loss}
        feat = self.rssm.get_feat(post)
        for name, head in self.heads.items():
            grad_head = (name in self.config.grad_heads)
            inp = feat if grad_head else tf.stop_gradient(feat)
            if name == 'unit_init_set':
                # to train initial unit set for the unit decoder, sample points for every unit in the batch and minimise
                unpadded_units = tf.where(tf.not_equal(data['unit_id'][..., 0], 1))  # find indices where unit type not 0
                total_units = tf.shape(unpadded_units)[0]

                # use our existing learned unit type embedding to convert the one-hot type and use that as label for training the prior
                unit_feats = tf.stop_gradient(self.encoder.get_unit_feats(data['unit_id'], data['unit_alliance'], data['unit_cloaked'], data['unit_continuous'], data['unit_binary']))  # dont train our embedding layer, we just want to sample from it
                unit_true = tf.gather_nd(unit_feats, unpadded_units)

                dist = head(total_units)
                like = tf.cast(dist.log_prob(unit_true), tf.float32)
                likes[name] = like
                losses[name] = -like.mean()
            elif name == 'units':
                unpadded_units = tf.cast(tf.not_equal(data['unit_id'][..., 0], 1), dtype=tf.int32)  # find indices where unit type not 0
                set_sizes = tf.reduce_sum(unpadded_units, axis=-1)

                initial_set = tf.stop_gradient(self.heads['unit_init_set'].sample(data['unit_id']))  # dont train the initial unit distribution when sampling a set

                unit_dists = head(initial_set, inp, set_sizes)

                like = tf.cast(prob_chamfer_distance(unit_dists['unit_id'], tf.cast(data['unit_id'], tf.int32),
                                                     unit_dists['unit_alliance'], tf.cast(data['unit_alliance'], tf.int32),
                                                     unit_dists['unit_cloaked'], tf.cast(data['unit_cloaked'], tf.int32),
                                                     unit_dists['unit_continuous'], tf.cast(data['unit_continuous'], tf.float32),
                                                     unit_dists['unit_binary'], tf.cast(data['unit_binary'], tf.int32),
                                                     set_sizes, self.config.max_units), tf.float32)
                likes[name] = like
                losses[name] = -like.mean()
            elif name == 'screen':
                out_probs = head(inp)
                for k in out_probs:
                    like = tf.cast(out_probs[k].log_prob(tf.cast(data[k], out_probs[k].dtype)), tf.float32)
                    likes[k] = like
                    losses[k] = -like.mean()
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
            state, _, _, _ = prev
            feat = self.rssm.get_feat(state)
            norm_policy, onehot_policy = action_policy(tf.stop_gradient(feat))
            action_prob = tf.nn.softmax(norm_policy.sample())

            # only take imagined available actions
            available_action_types = tf.cast(tf.stop_gradient(self.heads['available_actions'](feat).mode()), tf.float32)  # dont train the world model while imagining
            masked_actions = action_prob * available_action_types + available_action_types  # scale unavailable actions to 0, then available actions get a +1 bonus so that if everything is 0, they are still higher
            chosen_action_id = tf.argmax(masked_actions, axis=-1, output_type=tf.int32)
            chosen_action_oh = tf.one_hot(chosen_action_id, depth=tf.shape(action_prob)[-1])

            feat_action = tf.concat([feat, tf.cast(chosen_action_oh, feat.dtype)], -1)
            arg_policy_out = arg_policy(tf.stop_gradient(feat_action))  # dont train action policy while updating arg policy
            args = {}
            for a in self.arg_keys:
                args[a] = arg_policy_out[a].sample()

            action_vec = self.action_preprocess(chosen_action_oh, args)

            succ = self.rssm.img_step(state, action_vec)

            return succ, feat, action_prob, args

        feat = 0 * self.rssm.get_feat(start)

        init_norm_policy, init_onehot_policy = action_policy(feat)
        init_action = tf.nn.softmax(init_norm_policy.mode())
        init_available_action_types = tf.cast(tf.stop_gradient(self.heads['available_actions'](feat).mode()), tf.float32)  # dont train avl action head while learning policy
        init_masked_actions = init_action * init_available_action_types + init_available_action_types  # scale unavailable actions to 0, then available actions get a +1 bonus so that if everything is 0, they are still higher
        init_chosen_action = tf.one_hot(tf.argmax(init_masked_actions, axis=-1), depth=tf.shape(init_action)[-1])

        init_feat_action = tf.concat([feat, tf.cast(init_chosen_action, feat.dtype)], -1)
        init_arg_policy_out = arg_policy(init_feat_action)

        init_args = {}
        for a in self.arg_keys:
            init_args[a] = init_arg_policy_out[a].mode()

        succs, feats, actions, args = common.static_scan(step, tf.range(horizon), (start, feat, init_action, init_args))

        states = {k: tf.concat([start[k][None], v[:-1]], 0) for k, v in succs.items()}
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
        obs['screen_visibility'] = tf.one_hot(obs['screen'][..., 0], len(Visibility), dtype=dtype)  # screen visibility
        obs['screen_height'] = tf.cast(obs['screen'][..., 1:2], dtype=dtype) / 255.0 - 0.5  # screen height
        obs['screen_creep'] = tf.cast(obs['screen'][..., 2:3], dtype=dtype)  # creep / buildable / pathable
        obs['screen_buildable'] = tf.cast(obs['screen'][..., 3:4], dtype=dtype)
        obs['screen_pathable'] = tf.cast(obs['screen'][..., 4:5], dtype=dtype)
        obs['screen_effects'] = tf.one_hot(obs['screen'][..., 5], len(Effects), dtype=dtype)  # screen effects one-hot
        del obs['screen']

        # minimap preproc
        pp_mini_feat = []
        pp_mini_feat.append(tf.one_hot(obs['mini'][..., 0], len(Visibility), dtype=tf.float32))  # minimap visibility
        pp_mini_feat.append(tf.cast(obs['mini'][..., 1:2], dtype=tf.float32) / 255.0 - 0.5)  # minimap height
        pp_mini_feat.append(tf.one_hot(obs['mini'][..., 2], len(PlayerRelative), dtype=tf.float32))  # minimap player relative unit alliance
        pp_mini_feat.append(tf.cast(obs['mini'][..., 3:7], dtype=tf.float32) * 1.0)  # creep / buildable / pathable / camera
        obs['mini'] = tf.cast(tf.concat(pp_mini_feat, axis=-1), dtype=dtype)

        # unit preproc
        obs['unit_id'] = tf.one_hot(obs['units'][..., 0], len(set(get_unit_embed_lookup().values())), dtype=tf.int32)  # unit ids
        obs['unit_alliance'] = tf.one_hot(obs['units'][..., 1] - 1, 4, dtype=dtype)  # alliance: self = 1, ally = 2, neutral, enemy, -1 so its 0 indexed
        obs['unit_cloaked'] = tf.one_hot(obs['units'][..., 19] - 1, 4, dtype=dtype)  # cloak: Cloaked = 1, CloakedDetected = 2, NotCloaked = 3, Unknown = 4, -1 so its 0 indexed
        obs['unit_continuous'] = tf.concat([
            tf.cast(obs['units'][..., 2:5], dtype=dtype) / 255.0 - 0.5,  # health / shield / energy are all in scale 0-255
            tf.cast(obs['units'][..., 5:6], dtype=dtype) / float(self.config.screen_size) - 0.5,  # x pos
            tf.cast(obs['units'][..., 6:7], dtype=dtype) / float(self.config.screen_size) - 0.5,  # y pos
            tf.cast(obs['units'][..., 7:8], dtype=dtype) / 5.0 - 0.5,  # radius: biggest units (command centers) have radius of 5
            tf.cast(obs['units'][..., 10:11], dtype=dtype),  # build_progress
            tf.cast(obs['units'][..., 12:13], dtype=dtype) / 1800.0 - 0.5,  # mineral count
            tf.cast(obs['units'][..., 13:14], dtype=dtype) / 2250.0 - 0.5,  # vespene count
            tf.cast(obs['units'][..., 14:15], dtype=dtype) / 8.0 - 0.5,  # cargo taken
            tf.cast(obs['units'][..., 15:16], dtype=dtype) / 8.0 - 0.5,  # cargo max
            tf.cast(obs['units'][..., 21:22], dtype=dtype) / 3.0 - 0.5,  # attack upgrade
            tf.cast(obs['units'][..., 22:23], dtype=dtype) / 3.0 - 0.5,  # armour upgrade
            tf.cast(obs['units'][..., 23:24], dtype=dtype) / 3.0 - 0.5  # shield upgrade
        ], axis=-1)
        obs['unit_binary'] = tf.concat([
            tf.cast(obs['units'][..., 8:10], dtype=dtype),  # is_selected / is_blip
            tf.cast(obs['units'][..., 11:12], dtype=dtype),  # is_powered
            tf.cast(obs['units'][..., 16:19], dtype=dtype),  # is_flying / is_burrowed / is_in_cargo
            tf.cast(obs['units'][..., 20:21], dtype=dtype),  # is_hallucination
            tf.cast(obs['units'][..., 24:65], dtype=dtype)  # boolean buffs list
        ], axis=-1)
        del obs['units']

        # player preproc
        pp_player_features = []
        obs['player'] = tf.cast(obs['player'], dtype=tf.float32)
        pp_player_features.append(tf.sqrt(obs['player'][..., 0:1]) / 50)  # player minerals
        pp_player_features.append(tf.sqrt(obs['player'][..., 1:2]) / 50)  # player gas
        pp_player_features.append(obs['player'][..., 2:3] / 200)  # supply used
        pp_player_features.append(obs['player'][..., 3:4] / 200)  # supply max
        pp_player_features.append(obs['player'][..., 4:5] / 10)  # warp gates
        pp_player_features.append(obs['player'][..., 5:6] / 20)  # larva count
        obs['player'] = tf.cast(tf.concat(pp_player_features, axis=-1), dtype=dtype)

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
    def video_pred(self, data):
        data = self.preprocess(data)
        # truth_screen_visibility = data['screen_visibility'][:6]
        # truth_screen_height = data['screen_height'][:6]
        # truth_screen_creep = data['screen_creep'][:6]
        # truth_screen_buildable = data['screen_buildable'][:6]
        # truth_screen_pathable = data['screen_pathable'][:6]
        # truth_screen_effects = data['screen_effects'][:6]

        truth_unit_id = data['unit_id'][:6]
        truth_unit_alliance = data['unit_alliance'][:6]
        truth_unit_cloaked = data['unit_id'][:6]
        truth_unit_continuous = data['unit_id'][:6]
        truth_unit_binary = data['unit_binary'][:6]

        embed = self.encoder(data)
        full_action = self.action_preprocess(data['action_id'], data)
        states, _ = self.rssm.observe(embed[:6, :5], full_action[:6, :5])

        state_feat = self.rssm.get_feat(states)
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.rssm.imagine(full_action[:6, 5:], init)
        prior_feat = self.rssm.get_feat(prior)

        unpadded_units = tf.cast(tf.not_equal(truth_unit_id[..., 0], 1), dtype=tf.int32)  # find indices where unit type not 0
        set_sizes = tf.reduce_sum(unpadded_units, axis=-1)
        initial_set = self.heads['unit_init_set'].sample(truth_unit_id)

        unit_recon_dists = self.heads['units'](initial_set[:6, :5], state_feat, set_sizes[:6, :5])

        unit_openl_dists = self.heads['units'](initial_set[:6, 5:], prior_feat, set_sizes[:6, 5:])

        unit_recon = {}
        unit_openl = {}
        model = {}
        for u in unit_recon_dists:
            unit_recon = unit_recon_dists[u].mode()[:6]
            unit_openl = unit_openl_dists[u].mode()[:6]
            model[u] = tf.concat([unit_recon[:, :5], unit_openl], 1)


class Sc2ActorCritic(common.Module):

    def __init__(self, config, step, act_space, action_required_args):
        self.config = config
        self.step = step
        self.type_actor = common.Sc2MLP(act_space['action_id'].n, **config.type_actor)
        arg_space_sizes = {key: (value.n,) for (key, value) in act_space.items() if key != 'action_id'}
        self.arg_actor = common.Sc2CompositeMLP(arg_space_sizes, **config.arg_actor)
        self.critic = common.Sc2MLP([], **config.critic)
        self.action_required_args = action_required_args

        # create a lookup to find which actions each arg is used for, because its a pain to do action-to-arg in TF when iterating by arg types
        self.actions_using_arg = {}
        for arg in arg_space_sizes:
            acts = []
            for act in (range(act_space['action_id'].n)):
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

    def train(self, world_model, start, reward_fn):
        metrics = {}
        hor = self.config.imag_horizon
        with tf.GradientTape(persistent=True) as actor_tape:
            feat, state, action, action_args, disc = world_model.imagine(self.type_actor, self.arg_actor, start, hor)
            reward = reward_fn(feat)
            target, weight, mets1 = self.target(feat, action, action_args, reward, disc)

            type_actor_loss, mets2 = self.action_loss(feat, action, target, weight)
            arg_actor_loss, mets3 = self.arg_loss(feat, action, action_args, target, weight)

        with tf.GradientTape() as critic_tape:
            critic_loss, mets4 = self.critic_loss(feat, action, action_args, target, weight)

        metrics.update(self.type_actor_opt(actor_tape, type_actor_loss, self.type_actor))
        metrics.update(self.arg_actor_opt(actor_tape, arg_actor_loss, self.arg_actor))
        del actor_tape
        metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
        metrics.update(**mets1, **mets2, **mets3, **mets4)
        self.update_slow_target()  # Variables exist after first forward pass.
        return metrics

    @tf.function
    def action_loss(self, feat, action, target, weight):
        metrics = {}

        norm_policy, onehot_policy = self.type_actor(tf.stop_gradient(feat))

        if self.config.actor_grad == 'dynamics':
            objective = target
        elif self.config.actor_grad == 'reinforce':
            baseline = self.critic(feat[:-1]).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = onehot_policy.log_prob(action)[:-1] * advantage
        elif self.config.actor_grad == 'both':
            baseline = self.critic(feat[:-1]).mode()
            advantage = tf.stop_gradient(target - baseline)
            objective = onehot_policy.log_prob(action)[:-1] * advantage
            mix = common.schedule(self.config.actor_grad_mix, self.step)
            objective = mix * target + (1 - mix) * objective
            metrics[f'type_actor_grad_mix'] = mix
        else:
            raise NotImplementedError(self.config.actor_grad)
        ent = onehot_policy.entropy()
        ent_scale = common.schedule(self.config.actor_ent, self.step)
        objective += ent_scale * ent[:-1]
        actor_loss = -(weight[:-1] * objective).mean()
        metrics[f'type_actor_ent'] = ent.mean()
        metrics[f'type_actor_ent_scale'] = ent_scale
        return actor_loss, metrics

    @tf.function
    def arg_loss(self, feat, action, action_args, target, weight):
        metrics = {}
        chosen_action_id = tf.argmax(action, axis=-1, output_type=tf.int32)

        feat_action = tf.concat([feat, tf.cast(action, feat.dtype)], -1)
        policy = self.arg_actor(tf.stop_gradient(feat_action))

        arg_head_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        num_args_used = 0

        for arg_key in self.arg_keys:
            if self.config.actor_grad == 'dynamics':
                objective = target
            elif self.config.actor_grad == 'reinforce':
                baseline = self.critic(feat[:-1]).mode()
                advantage = tf.stop_gradient(target - baseline)
                objective = policy[arg_key].log_prob(action_args[arg_key])[:-1] * advantage
            elif self.config.actor_grad == 'both':
                baseline = self.critic(feat[:-1]).mode()
                advantage = tf.stop_gradient(target - baseline)
                objective = policy[arg_key].log_prob(action_args[arg_key])[:-1] * advantage
                mix = common.schedule(self.config.actor_grad_mix, self.step)
                objective = mix * target + (1 - mix) * objective
                metrics[f'{arg_key}_actor_grad_mix'] = mix
            else:
                raise NotImplementedError(self.config.actor_grad)

            ent = policy[arg_key].entropy()
            ent_scale = common.schedule(self.config.actor_ent, self.step)
            objective += ent_scale * ent[:-1]

            # # tile our chosen actions and check if this arg is required for each action
            actions_needing_arg = self.actions_using_arg[arg_key]
            actions_tiled = tf.tile(tf.expand_dims(chosen_action_id, -1), [1, 1, tf.size(actions_needing_arg)])
            needed = tf.where(tf.greater(tf.math.count_nonzero(tf.equal(actions_tiled, actions_needing_arg), axis=-1), 0))

            if tf.size(needed) > 0:
                # only calculate losses for used args
                weight_trimmed = weight[:-1]
                weight_trimmed = tf.gather_nd(weight_trimmed, needed)
                objective = tf.gather_nd(objective, needed)

                actor_loss = -tf.reduce_mean(weight_trimmed * objective, axis=0, keepdims=True)
                arg_head_losses = arg_head_losses.write(num_args_used, actor_loss)
                num_args_used += 1

            metrics[f'{arg_key}_actor_ent'] = ent.mean()
            metrics[f'{arg_key}_actor_ent_scale'] = ent_scale

        arg_losses = arg_head_losses.concat()
        arg_loss = tf.reduce_mean(arg_losses)

        return arg_loss, metrics

    def critic_loss(self, feat, action, action_args, target, weight):
        dist = self.critic(feat)[:-1]
        target = tf.stop_gradient(target)
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {f'critic': dist.mode().mean()}
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
                for s, d in zip(self.critic.variables, self._target_critic.variables):
                    d.assign(mix * s + (1 - mix) * d)
            self._updates.assign_add(1)
