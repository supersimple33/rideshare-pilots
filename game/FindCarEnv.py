from typing import Generic, Literal, TypeAlias, TypeVar

import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace, MultiDiscrete, OneOf as OneOfSpace
import numpy as np

from game.GridGeneration import NoObstaclesScheme, ObstacleGenerationScheme
from game.utils import Contents, Directions, PosInt, PositionList

NOOP_GENERATION_SCHEME = NoObstaclesScheme()

H = TypeVar("H", bound=PosInt)
W = TypeVar("W", bound=PosInt)

ObsType: TypeAlias = dict[str, MultiDiscrete]
ActionType: TypeAlias = Directions


class FindCarEnv(gym.Env[ObsType, ActionType], Generic[H, W]):
    """A mini grid environment. It can generate a different layout each time according to
    the generation method specified. The objective is to move an agent to the correct car
    which will all look identical to the agent until it is very close to it."""

    width: W
    height: H
    obstacle_scheme: ObstacleGenerationScheme
    grid: np.ndarray[tuple[W, H], np.dtype[np.uint8]]
    _target_location: np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
    _agent_location: np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]

    def __init__(
        self,
        width: W,
        height: H,
        obstacle_scheme: ObstacleGenerationScheme = NOOP_GENERATION_SCHEME,
    ):
        self.obstacle_scheme = obstacle_scheme
        self.width = width
        self.height = height

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = DictSpace(
            {
                "agent_position": MultiDiscrete([width, height], dtype=np.int32),
                "target_position": MultiDiscrete([width, height], dtype=np.int32),
                "board": MultiDiscrete(3 * np.ones((width, height), dtype=np.uint8)),
            }
        )
        # [0,1] , [0,-1]. [1,0], [-1,0]
        self.action_space = OneOfSpace(
            [
                MultiDiscrete(Directions.UP.value),  # type: ignore
                MultiDiscrete(Directions.DOWN.value),  # type: ignore
                MultiDiscrete(Directions.RIGHT.value),  # type: ignore
                MultiDiscrete(Directions.LEFT.value),  # type: ignore
            ]
        )

        self.grid = np.empty((width, height), dtype=np.int8)  # type: ignore

