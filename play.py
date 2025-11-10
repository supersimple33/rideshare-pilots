from gymnasium.utils.play import play

from game.FindCarEnv import FindCarEnv
from game.GridGeneration import DisjointBlobs
from game.utils import Direction

KEYS_TO_ACTION = {
    "w": Direction.UP,
    "s": Direction.DOWN,
    "a": Direction.LEFT,
    "d": Direction.RIGHT,
}

obstacle_gen = DisjointBlobs(
    max_count=2,
    max_size=10,
)
env = FindCarEnv(
    width=10,
    height=6,
    num_fake_targets=2,
    render_mode="rgb_array",
    obstacle_scheme=obstacle_gen,
)

play(env, keys_to_action=KEYS_TO_ACTION, wait_on_player=True, fps=10)
