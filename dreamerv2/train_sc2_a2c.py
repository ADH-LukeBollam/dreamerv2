import collections
import functools
import logging
import os
import pathlib
import sys
import warnings
import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import agent_a2c
import elements
import common

configs = pathlib.Path(sys.argv[0]).parent / 'configs_sc2.yaml'
configs = yaml.safe_load(configs.read_text())
config = elements.Config(configs['defaults'])
parsed, remaining = elements.FlagParser(configs=['defaults']).parse_known(
    exit_on_help=False)
for name in parsed.configs:
    config = config.update(configs[name])
config = elements.FlagParser(config).parse(remaining)
logdir = pathlib.Path(config.logdir).expanduser()
config = config.update(
    steps=config.steps // config.action_repeat,
    eval_every=config.eval_every // config.action_repeat,
    log_every=config.log_every // config.action_repeat,
    time_limit=config.time_limit // config.action_repeat,
    prefill=config.prefill // config.action_repeat)

tf.config.experimental_run_functions_eagerly(not config.jit)
message = 'No GPU found. To actually train on CPU remove this assert.'
assert tf.config.experimental.list_physical_devices('GPU'), message
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
assert config.precision in (16, 32), config.precision
if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec

    prec.set_policy(prec.Policy('mixed_float16'))

print('Logdir', logdir)
step = elements.Counter(0)
outputs = [
    elements.TerminalOutput(),
    elements.JSONLOutput(logdir),
    elements.TensorBoardOutput(logdir, 1),
]
logger = elements.Logger(step, outputs, multiplier=config.action_repeat)
metrics = collections.defaultdict(list)
should_train = elements.Every(config.train_every)
should_log = elements.Every(config.log_every)
should_video_train = elements.Every(config.eval_every)
should_video_eval = elements.Every(config.eval_every)

def make_env(mode):
    suite, task = config.task.split('_', 1)
    env = common.Sc2(task, config.screen_size, config.mini_size, config.max_units, 22, 0, False, False)
    env = common.OneHotAction(env)
    env = common.TimeLimit(env, config.time_limit)
    env = common.RewardObs(env)
    env = common.ResetObs(env)
    return env


def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    logger.write()


print('Create envs.')
train_env = make_env('train')
eval_env = make_env('eval')
action_space = train_env.action_space
action_req_args = train_env.action_arg_lookup
train_driver = common.A2CDriver(train_env)
train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
train_driver.on_step(lambda _: step.increment())
eval_driver = common.A2CDriver(eval_env)
eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))

print('Create agent.')
agnt = agent_a2c.A2CAgent(config, logger, action_space, step, action_req_args, train_env)
if (logdir / 'variables.pkl').exists():
    agnt.load(logdir / 'variables.pkl')


def train_step(mets):
    if should_log(step):
        for _ in range(config.train_steps):
            [metrics[key].append(value) for key, value in mets.items()]
        for name, values in metrics.items():
            logger.scalar(name, np.array(values, np.float64).mean())
            metrics[name].clear()


train_driver.on_step(train_step)

while step < config.steps:
    logger.write()
    print('Start evaluation.')
    eval_driver(agnt.eval, episodes=config.eval_eps)
    print('Start training.')
    train_driver(agnt.train, steps=config.eval_every)
    agnt.save(logdir / 'variables.pkl')

for env in [train_env, eval_env]:
    try:
        env.close()
    except Exception:
        pass
