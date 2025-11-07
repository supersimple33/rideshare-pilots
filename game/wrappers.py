from typing import Generic, TypeVar, TypedDict

from gymnasium.spaces import Dict as DictSpace, MultiDiscrete
import numpy as np

from game.FindCarEnv import ObsType, H, W
from game.utils import Content, Location, PosInt, content_to_one_hot, window_at

N = TypeVar("N", bound=PosInt)


class OneHotObsType(TypedDict, Generic[H, W]):
    agent_position: Location
    board: np.ndarray[tuple[W, H, PosInt], np.dtype[np.uint8]]


def local_view_wrapper(
    obs_space: DictSpace, n: N  # pyright: ignore[reportInvalidTypeVarUse]
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

    board_space: MultiDiscrete = obs_space["board"]  # type: ignore
    width, height = board_space.shape
    obs_space = DictSpace(
        {
            "agent_position": MultiDiscrete([width, height], dtype=np.int32),
            "board": MultiDiscrete(
                np.full((n, n), len(Content)),
                dtype=np.uint8,
            ),
        }
    )

    return wrapper, obs_space


def one_hot_wrapper(obs_space: DictSpace):  # pyright: ignore[reportInvalidTypeVarUse]
    """Gets a wrapper that converts observations to one-hot encoding."""

    def wrapper(observation: ObsType[H, W]) -> OneHotObsType[H, W]:
        """A wrapper to convert observations to one-hot encoding."""
        return {
            "agent_position": observation["agent_position"],
            "board": content_to_one_hot(observation["board"]),
        }

    board_space: MultiDiscrete = obs_space["board"]  # type: ignore
    width, height = board_space.shape
    obs_space = DictSpace(
        {
            "agent_position": MultiDiscrete([width, height], dtype=np.int32),
            "board": MultiDiscrete(
                np.full((width, height, len(Content)), 2),
                dtype=np.uint8,
            ),
        }
    )

    return wrapper, obs_space
