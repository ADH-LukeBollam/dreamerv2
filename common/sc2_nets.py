import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as prec
import common
from models.set_transformer import TransformerLayer
from models.unit_encoder import UnitEncoder
from pysc2.lib.features import Visibility, Effects
from pysc2.lib.units import get_unit_embed_lookup


class Sc2RSSM(common.Module):

    def __init__(self, stoch=30, deter=200, hidden=200, discrete=False, act=tf.nn.elu, std_act='softplus', min_std=0.1):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._std_act = std_act
        self._min_std = min_std
        self._cell = Sc2GRUCell(self._deter, norm=True)
        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        if self._discrete:
            state = dict(
                logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        else:
            state = dict(
                mean=tf.zeros([batch_size, self._stoch], dtype),
                std=tf.zeros([batch_size, self._stoch], dtype),
                stoch=tf.zeros([batch_size, self._stoch], dtype),
                deter=self._cell.get_initial_state(None, batch_size, dtype))
        return state

    @tf.function
    def observe(self, embed, action_input, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action_input)[0])
        embed, action_input = swap(embed), swap(action_input)
        post, prior = common.static_scan(lambda prev, inputs: self.obs_step(prev[0], *inputs), (action_input, embed), (state, state))
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    @tf.function
    def imagine(self, action_input, state=None):
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if state is None:
            state = self.initial(tf.shape(action_input)[0])
        assert isinstance(state, dict), state
        action = swap(action_input)
        prior = common.static_scan(self.img_step, action, state)
        prior = {k: swap(v) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        stoch = self._cast(state['stoch'])
        if self._discrete:
            shape = stoch.shape[:-2] + [self._stoch * self._discrete]
            stoch = tf.reshape(stoch, shape)
        return tf.concat([stoch, state['deter']], -1)

    def get_dist(self, state):
        if self._discrete:
            logit = state['logit']
            logit = tf.cast(logit, tf.float32)
            dist = tfd.Independent(common.OneHotDist(logit), 1)
        else:
            mean, std = state['mean'], state['std']
            mean = tf.cast(mean, tf.float32)
            std = tf.cast(std, tf.float32)
            dist = tfd.MultivariateNormalDiag(mean, std)
        return dist

    @tf.function
    def obs_step(self, prev_state, prev_action, embed, sample=True):
        prior = self.img_step(prev_state, prev_action, sample)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs_out', tfkl.Dense, self._hidden, self._act)(x)
        stats = self._suff_stats_layer('obs_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        post = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = self._cast(prev_state['stoch'])
        prev_action = self._cast(prev_action)
        if self._discrete:
            shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
            prev_stoch = tf.reshape(prev_stoch, shape)
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get('img_in', tfkl.Dense, self._hidden, self._act)(x)
        deter = prev_state['deter']
        x, deter = self._cell(x, [deter])
        deter = deter[0]  # Keras wraps the state in a list.
        x = self.get('img_out', tfkl.Dense, self._hidden, self._act)(x)
        stats = self._suff_stats_layer('img_dist', x)
        dist = self.get_dist(stats)
        stoch = dist.sample() if sample else dist.mode()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def _suff_stats_layer(self, name, x):
        if self._discrete:
            x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
            logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
            return {'logit': logit}
        else:
            x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
            mean, std = tf.split(x, 2, -1)
            std = {
                'softplus': lambda: tf.nn.softplus(std),
                'sigmoid': lambda: tf.nn.sigmoid(std),
                'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
            }[self._std_act]()
            std = std + self._min_std
            return {'mean': mean, 'std': std}

    def kl_loss(self, post, prior, forward, balance, free, free_avg):
        kld = tfd.kl_divergence
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = kld(self.get_dist(lhs), self.get_dist(rhs))
            loss = tf.maximum(value, free).mean()
        else:
            value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
            value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
            if free_avg:
                loss_lhs = tf.maximum(value_lhs.mean(), free)
                loss_rhs = tf.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = tf.maximum(value_lhs, free).mean()
                loss_rhs = tf.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


class Sc2Encoder(common.Module):

    def __init__(self, act=tf.nn.elu,
                 avl_action_widths=(16, 16),
                 screen_depth=32, screen_kernels=(4, 4, 4, 4),
                 minimap_depth=32, minimap_kernels=(4, 4, 4, 4),
                 player_widths=(8, 8),
                 unit_embed_dim=16, unit_pp_dim=64, unit_num_layers=2, unit_trans_dim=256, unit_trans_heads=4,
                 mixing_widths=(512,)):
        self._avl_action_encoder = Sc2MlpEncoder('avl_action', act, avl_action_widths)
        self._screen_encoder = Sc2ConvEncoder('screen', screen_depth, act, screen_kernels)
        self._minimap_encoder = Sc2ConvEncoder('minimap', minimap_depth, act, minimap_kernels)
        self._player_encoder = Sc2MlpEncoder('player', act, player_widths)
        self._unit_encoder = UnitEncoder(unit_pp_dim, unit_num_layers, unit_trans_dim, unit_trans_heads)
        self._mixing_encoder = Sc2MlpEncoder('mixing', act, mixing_widths)

        num_unit_types = len(set(get_unit_embed_lookup().values()))
        self.unit_type_embedding = tf.keras.layers.Embedding(num_unit_types, unit_embed_dim)

    @tf.function
    def __call__(self, obs):
        avl_actions = self._avl_action_encoder(obs['available_actions'])

        screen_obs = tf.concat([obs['screen_visibility'], obs['screen_height'], obs['screen_creep'], obs['screen_buildable'], obs['screen_pathable'], obs['screen_effects']], axis=-1)
        screen = self._screen_encoder(screen_obs)

        minimap = self._minimap_encoder(obs['mini'])
        player = self._player_encoder(obs['player'])

        unit_feats = self.get_unit_feats(obs['unit_id'], obs['unit_alliance'], obs['unit_cloaked'], obs['unit_continuous'], obs['unit_binary'])
        units = self._unit_encoder(unit_feats)

        state_embed = tf.concat([avl_actions, screen, minimap, player, units], -1)
        mixed = self._mixing_encoder(state_embed)

        return mixed

    @tf.function
    def get_unit_feats(self, unit_ids_onehot, unit_alliance, unit_cloaked, unit_continuous, unit_binary):
        unit_ids = tf.argmax(unit_ids_onehot, axis=-1)
        embedded = self.unit_type_embedding(unit_ids)

        unit_attributes = tf.concat([unit_alliance, unit_cloaked, unit_continuous, unit_binary], axis=-1)

        return tf.concat([embedded, unit_attributes], axis=-1)


class Sc2MlpEncoder(common.Module):
    def __init__(self, name_prefix, act=tf.nn.elu, widths=(16, 16)):
        self._name_prefix = name_prefix
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._widths = widths

    @tf.function
    def __call__(self, inputs):
        x = inputs
        for i, width in enumerate(self._widths):
            x = self._act(self.get(f'{self._name_prefix}_h{i}', tfkl.Dense, width)(x))
        return x


class Sc2ConvEncoder(common.Module):
    def __init__(self, name_prefix, depth, act, kernels):
        self._name_prefix = name_prefix
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._depth = depth
        self._kernels = kernels

    @tf.function
    def __call__(self, inputs):
        x = tf.reshape(inputs, (-1,) + tuple(inputs.shape[-3:]))
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** i * self._depth
            x = self._act(self.get(f'{self._name_prefix}_h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
        x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
        shape = tf.concat([tf.shape(inputs)[:-3], [x.shape[-1]]], 0)
        return tf.reshape(x, shape)


class Sc2ScreenDecoder(common.Module):

    def __init__(self, screen_size, depth=32, act=tf.nn.elu, kernels=(5, 5, 6, 6)):
        self._screen_shape = (screen_size, screen_size)
        self._depth = depth
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._kernels = kernels

    def __call__(self, features):
        ConvT = tfkl.Conv2DTranspose
        x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
        for i, kernel in enumerate(self._kernels):
            depth = 2 ** (len(self._kernels) - i - 2) * self._depth
            act = self._act
            x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)

        processed = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._screen_shape, (int(depth),)], 0))

        outs = {}

        outs['screen_visibility'] = self.get('screen_visibility_out', Sc2DistLayer, (len(Visibility),), 'onehot_batch', dist_shape=self._screen_shape + (len(Visibility),))(processed)
        outs['screen_height'] = self.get('screen_height_out', Sc2DistLayer, (1,), 'mse', dist_shape=self._screen_shape + (1,))(processed)
        outs['screen_creep'] = self.get('screen_creep_out', Sc2DistLayer, (1,), 'binary', dist_shape=self._screen_shape + (1,))(processed)
        outs['screen_buildable'] = self.get('screen_buildable_out', Sc2DistLayer, (1,), 'binary', dist_shape=self._screen_shape + (1,))(processed)
        outs['screen_pathable'] = self.get('screen_pathable_out', Sc2DistLayer, (1,), 'binary', dist_shape=self._screen_shape + (1,))(processed)
        outs['screen_effects'] = self.get('screen_effects_out', Sc2DistLayer, (len(Effects),), 'binary', dist_shape=self._screen_shape + (len(Effects),))(processed)

        return outs


class UnitDecoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, trans_dim, trans_heads):
        super(UnitDecoder, self).__init__()
        # process initial set to transformer dimension
        self.embedding = tf.keras.layers.Conv1D(trans_dim, 1, kernel_initializer='glorot_uniform', use_bias=True,
                                                bias_initializer=tf.constant_initializer(0.1))\

        self.num_layers = num_layers
        self.trans_dim = trans_dim
        self.transformer = [TransformerLayer(trans_dim, trans_heads) for _ in range(num_layers)]
        self.num_unit_types = len(set(get_unit_embed_lookup().values()))

    def call(self, initial_set, encoding, sizes):
        # flatten our batch + timestep dimensions together
        x = tf.reshape(initial_set, (-1,) + tuple(initial_set.shape[-2:]))
        x_encoding = tf.reshape(encoding, (-1, 1) + tuple(encoding.shape[-1:]))

        set_size = tf.shape(x)[1]     # batch, set, features

        # concat the encoding vector onto each initial set element
        encoded_shaped = tf.tile(x_encoding, [1, set_size, 1])
        conditioned_initial_set = tf.concat([x, encoded_shaped], 2)

        mask = tf.reshape(tf.cast(tf.math.logical_not(tf.sequence_mask(sizes, set_size)), tf.float32), [-1, 1, 1, set_size])

        x = self.embedding(conditioned_initial_set)

        for i in range(self.num_layers):
            x = self.transformer[i](x, x, mask)

        processed = tf.reshape(x, tf.concat([tf.shape(initial_set)[:-1], (self.trans_dim,)], 0))

        outs = {}

        outs['unit_id'] = self.get('unit_id_out', Sc2DistLayer, (self.num_unit_types,), 'onehot_batch')(processed)
        outs['unit_alliance'] = self.get('unit_alliance_out', Sc2DistLayer, (4,), 'onehot_batch')(processed)
        outs['unit_cloaked'] = self.get('unit_cloaked_out', Sc2DistLayer, (4,), 'onehot_batch')(processed)
        outs['unit_continuous'] = self.get('unit_continuous_out', Sc2DistLayer, (13,), 'mse')(processed)
        outs['unit_binary'] = self.get('unit_binary_out', Sc2DistLayer, (49,), 'binary')(processed)

        return outs


class Sc2MLP(common.Module):
    def __init__(self, shape, layers, units, act=tf.nn.elu, **out):
        self._shape = (shape,) if isinstance(shape, int) else shape
        self._layers = layers
        self._units = units
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        return self.get('out', Sc2DistLayer, self._shape, **self._out)(x)


class Sc2CompositeMLP(common.Module):

    def __init__(self, shapes_dict, layers, units, act=tf.nn.elu, **out):
        self._shapes_dict = shapes_dict   # lookup of shapes, eg. ['size_0': [16], 'size_1': [16], 'colour': [2] ...]
        self._layers = layers
        self._units = units
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._out = out

    def __call__(self, features):
        x = tf.cast(features, prec.global_policy().compute_dtype)
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)

        out = {}
        for k in self._shapes_dict.keys():
            out[k] = self.get(f'out_{k}', Sc2DistLayer, self._shapes_dict[k], **self._out)(x)

        return out


class Sc2GRUCell(tf.keras.layers.AbstractRNNCell):

    def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
        super().__init__()
        self._size = size
        self._act = getattr(tf.nn, act) if isinstance(act, str) else act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
        if norm:
            self._norm = tfkl.LayerNormalization(dtype=tf.float32)

    @property
    def state_size(self):
        return self._size

    @tf.function
    def call(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(tf.concat([inputs, state], -1))
        if self._norm:
            dtype = parts.dtype
            parts = tf.cast(parts, tf.float32)
            parts = self._norm(parts)
            parts = tf.cast(parts, dtype)
        reset, cand, update = tf.split(parts, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = self._act(reset * cand)
        update = tf.nn.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Sc2DistLayer(common.Module):

    def __init__(self, shape, dist='mse', min_std=0.1, init_std=0.0, dist_shape=None):
        self._shape = shape
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        if dist_shape is None:
            self._dist_shape = self._shape
        else:
            self._dist_shape = dist_shape

    def __call__(self, inputs):
        out = self.get('out', tfkl.Dense, np.prod(self._shape))(inputs)
        out = tf.reshape(out, tf.concat([tf.shape(inputs)[:-1], self._shape], 0))
        out = tf.cast(out, tf.float32)
        if self._dist in ('normal', 'tanh_normal', 'trunc_normal', 'normal_onehot'):
            std = self.get('std', tfkl.Dense, np.prod(self._dist_shape))(inputs)
            std = tf.reshape(std, tf.concat([tf.shape(inputs)[:-1], self._dist_shape], 0))
            std = tf.cast(std, tf.float32)
        if self._dist == 'mse':
            dist = tfd.Normal(out, 1.0)
            return tfd.Independent(dist, len(self._dist_shape))
        if self._dist == 'normal':
            dist = tfd.Normal(out, std)
            return tfd.Independent(dist, len(self._dist_shape))
        if self._dist == 'binary':
            dist = tfd.Bernoulli(out)
            return tfd.Independent(dist, len(self._dist_shape))
        if self._dist == 'tanh_normal':
            mean = 5 * tf.tanh(out / 5)
            std = tf.nn.softplus(std + self._init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, common.TanhBijector())
            dist = tfd.Independent(dist, len(self._dist_shape))
            return common.SampleDist(dist)
        if self._dist == 'trunc_normal':
            std = 2 * tf.nn.sigmoid((std + self._init_std) / 2) + self._min_std
            dist = common.TruncNormalDist(tf.tanh(out), std, -1, 1)
            return tfd.Independent(dist, 1)
        if self._dist == 'onehot':
            return common.OneHotDist(out)
        if self._dist == 'onehot_batch':
            onehot = common.OneHotDist(out)
            return tfd.Independent(onehot, len(self._dist_shape) - 1)
        if self._dist == 'normal_onehot':
            dist = tfd.Normal(out, std)
            norm = tfd.Independent(dist, len(self._dist_shape))
            onehot = common.OneHotDist(out)
            return norm, onehot
        NotImplementedError(self._dist)
