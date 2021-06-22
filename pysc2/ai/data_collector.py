from pysc2.agents import random_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app
from pysc2 import maps
from absl import app
from absl import flags
import threading
from pysc2.env import available_actions_printer
import time
from pysc2.lib import point_flag
from typing import List, Dict
from pysc2.lib.features import ScreenFeatures, FeatureUnit, Player, EffectPos
from pysc2.env.environment import TimeStep
from pysc2.lib.named_array import NamedNumpyArray
from pysc2.lib.actions import FunctionCall
import pickle



FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")
point_flag.DEFINE_point("feature_screen_size", "32", "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "32", "Resolution for minimap feature layers.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_bool("render", True, "Whether to render with pygame.")
flags.DEFINE_integer("max_episodes", 1, "Total episodes.")
flags.DEFINE_string("map", None, "Name of a map to use.")
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")


class StaticMap:
    def __init__(self, game_map: ScreenFeatures):
        self.height_map = game_map.height_map
        self.buildable = game_map.buildable
        self.pathable = game_map.pathable


class GameHistory:
    def __init__(self, game_map: ScreenFeatures):
        self._static_map: StaticMap = StaticMap(game_map)
        self._world_state_history: List[WorldState] = []
        self._action_history: List[FunctionCall] = []

    def add_action(self, action: FunctionCall):
        self._action_history.append(action)

    def add_state(self, timestep: TimeStep):
        state = WorldState(timestep.observation.feature_screen,
                           timestep.observation.raw_units,
                           timestep.observation.raw_effects,
                           timestep.observation.player)
        self._world_state_history.append(state)


class WorldState:
    def __init__(self, screen_state: ScreenFeatures, unit_state: NamedNumpyArray, effect_state: NamedNumpyArray, player_state:
    Player):
        self._screen_state: Dict = self.collect_relevant_screen(screen_state)
        self._unit_state: NamedNumpyArray[FeatureUnit] = unit_state
        self._effect_state: NamedNumpyArray[EffectPos] = effect_state
        self._structured_state: Player = player_state

    def collect_relevant_screen(self, screen: ScreenFeatures):
        screen_state = {
            "visibility_map": screen["visibility_map"],
            "creep": screen["creep"],
            "power": screen["power"]
        }
        return screen_state

def run_loop(agent, env, max_frames=0, max_episodes=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    agent.setup(observation_spec[0], action_spec[0])

    try:
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            timestep = env.reset()
            game_history = GameHistory(timestep[0].observation.feature_screen)
            game_history.add_state(timestep[0])

            agent.reset()
            while True:
                total_frames += 1
                action = agent.step(timestep[0])
                game_history.add_action(action)
                if max_frames and total_frames >= max_frames:
                    return
                if timestep[0].last():
                    break
                timestep = env.step([action])
                game_history.add_state(timestep[0])

            with open('C:\\temp\\test.pkl', 'wb') as output:
                pickle.dump(game_history, output, pickle.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))


def run_thread(agent_class, players, map_name, visualize):
    """Run one thread worth of the environment with agents."""

    # no need to set action space if only 1 of raw/feature screen has been chosen
    with sc2_env.SC2Env(
        map_name=map_name,
        battle_net_map=False,
        players=players,
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=FLAGS.feature_screen_size,
            feature_minimap=FLAGS.feature_minimap_size,
            rgb_screen=None,
            rgb_minimap=None,
            action_space=None,
            use_feature_units=False,
            use_raw_units=True),
        step_mul=FLAGS.step_mul,
        game_steps_per_episode=None,
        disable_fog=False,
        visualize=visualize,
        realtime=False) as env:
        env = available_actions_printer.AvailableActionsPrinter(env)
        agent = agent_class()
        run_loop(agent, env, 0, FLAGS.max_episodes)
        if FLAGS.save_replay:
            env.save_replay(agent_class.__name__)


def main(unused_argv):
    """Run an agent."""
    players = []
    players.append(sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy, sc2_env.BotBuild.random))

    agent_class = random_agent.RandomAgent
    players.append(sc2_env.Agent(sc2_env.Race.zerg, 'RandomAgent'))

    threads = []
    for _ in range(FLAGS.parallel - 1):
        t = threading.Thread(target=run_thread, args=(agent_class, players, FLAGS.map, False))
        threads.append(t)
        t.start()

    run_thread(agent_class, players, FLAGS.map, FLAGS.render)

    for t in threads:
        t.join()


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)
