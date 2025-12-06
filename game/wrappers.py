from typing import Generic, TypeVar, TypedDict

from gymnasium.spaces import Dict as DictSpace, MultiDiscrete, MultiBinary
import numpy as np

from game.FindCarEnv import ObsType, H, W
from game.utils import (
    Content,
    Location,
    PosInt,
    content_to_one_hot,
    obscure_cars,
    reachable_window_at,
)

N = TypeVar("N", bound=PosInt)


class OneHotObsType(TypedDict, Generic[H, W]):
    agent_position: Location
    board: np.ndarray[tuple[H, W, PosInt], np.dtype[np.uint8]]

    def __annotate__(self) -> dict:  # type: ignore
        """Annotate the shape of the board for static type checkers."""
        return {}  # type: ignore


def local_view_wrapper(
    obs_space: DictSpace, n: N  # pyright: ignore[reportInvalidTypeVarUse]
):
    """Gets a wrapper that converts observations to a local view of NxN centered on the agent."""

    def wrapper(observation: ObsType[H, W]) -> ObsType[N, N]:
        """A wrapper to convert observations to a local view."""

        return {
            "agent_position": observation["agent_position"],
            "board": reachable_window_at(
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


def car_hider_wrapper(obs_space: DictSpace, n: PosInt):
    """Gets a wrapper that obscures whether a car is a target or not until within n distance. Unless a helper is within view."""

    def wrapper(observation: ObsType[PosInt, PosInt]) -> ObsType[PosInt, PosInt]:
        """A wrapper to convert observations to a local view."""

        return {
            "agent_position": observation["agent_position"],
            "board": obscure_cars(
                observation["board"],
                n,
            ),
        }

    return wrapper, obs_space


def one_hot_wrapper(obs_space: DictSpace):  # pyright: ignore[reportInvalidTypeVarUse]
    """Gets a wrapper that converts observations to one-hot encoding."""

    def wrapper(
        observation: ObsType[H, W],
    ) -> np.ndarray[tuple[H, W, PosInt], np.dtype[np.uint8]]:
        """A wrapper to convert observations to one-hot encoding."""
        return content_to_one_hot(observation["board"])

    board_space: MultiDiscrete = obs_space["board"]  # type: ignore
    width, height = board_space.shape
    new_obs = MultiBinary((width, height, len(Content)))

    return wrapper, new_obs
