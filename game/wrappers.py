from typing import TypeVar

from gymnasium.spaces import Dict as DictSpace, MultiDiscrete
from gymnasium.wrappers import TransformObservation as TransformObservationWrapper
import numpy as np

from game.FindCarEnv import ObsType, FindCarEnv, H, W
from game.utils import Content, PosInt, window_at

N = TypeVar("N", bound=PosInt)


def local_view_wrapper(
    env: FindCarEnv[H, W], n: N  # pyright: ignore[reportInvalidTypeVarUse]
):
    """Gets a wrapper that converts observations to a local view of NxN centered on the agent."""

    def wrapper(observation: ObsType[H, W]) -> ObsType[N, N]:
        """A wrapper to convert observations to a local view."""

        return {
            "agent_position": observation["agent_position"],
            "board": window_at(
                observation["board"],
                observation["agent_position"],
                n,
                pad_value=Content.Border,
            ),
        }

    obs_space = DictSpace(
        {
            "agent_position": MultiDiscrete([env.width, env.height], dtype=np.int32),
            "board": MultiDiscrete(
                np.full((n, n), len(Content)),
                dtype=np.uint8,
            ),
        }
    )

    return TransformObservationWrapper(env, wrapper, obs_space)
