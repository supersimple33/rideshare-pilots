from typing import Any, Generic, TypeAlias, TypeVar, TypedDict

import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace, MultiDiscrete, OneOf as OneOfSpace
import numpy as np

from game.GridGeneration import NoObstaclesScheme, ObstacleGenerationScheme
from game.utils import Contents, Direction, Location, PosInt, PositionList

NOOP_GENERATION_SCHEME = NoObstaclesScheme()

H = TypeVar("H", bound=PosInt)
W = TypeVar("W", bound=PosInt)


class ObsType(TypedDict, Generic[H, W]):
    agent_position: Location
    target_position: Location
    board: np.ndarray[tuple[W, H], np.dtype[np.uint8]]


ActionType: TypeAlias = Direction


class FindCarEnv(gym.Env[ObsType[H, W], ActionType], Generic[H, W]):
    """A mini grid environment. It can generate a different layout each time according to
    the generation method specified. The objective is to move an agent to the correct car
    which will all look identical to the agent until it is very close to it."""

    width: W
    height: H
    obstacle_scheme: ObstacleGenerationScheme
    grid: np.ndarray[tuple[W, H], np.dtype[np.uint8]]
    _target_location: Location
    _agent_location: Location

    def __init__(
        self,
        width: W,
        height: H,
        obstacle_scheme: ObstacleGenerationScheme = NOOP_GENERATION_SCHEME,
    ):
        """Initialize the FindCar environment.

        Args:
            width: The width of the grid.
            height: The height of the grid.
            obstacle_scheme: The obstacle generation scheme to use when generating the grid. Defaults to no obstacles.
        """
        self.obstacle_scheme = obstacle_scheme
        self.width = width
        self.height = height

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = DictSpace(
            {
                "agent_position": MultiDiscrete([width, height], dtype=np.int32),
                "target_position": MultiDiscrete([width, height], dtype=np.int32),
                "board": MultiDiscrete(
                    np.ones_like((width, height), dtype=np.int32) * len(Contents),
                    dtype=np.uint8,
                ),
            }
        )  # type: ignore
        self.action_space = OneOfSpace(
            [
                MultiDiscrete(Direction.UP),  # type: ignore
                MultiDiscrete(Direction.DOWN),  # type: ignore
                MultiDiscrete(Direction.RIGHT),  # type: ignore
                MultiDiscrete(Direction.LEFT),  # type: ignore
            ]
        )

        self.grid = np.empty((width, height), dtype=np.int8)  # type: ignore

    def view(self) -> ObsType[H, W]:
        """Get the current observation of the environment.

        Returns:
            The current observation as a dictionary.
        """
        return {
            "agent_position": self._agent_location,
            "target_position": self._target_location,
            "board": self.grid,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType[H, W], dict[str, None]]:
        """Reset the environment to an initial state.

        Args:
            seed: An optional seed for the random number generator.
            options: Additional options for resetting the environment.
        Returns:
            A tuple containing the initial observation and a blank dictionary.
        """
        super().reset(seed=seed, options=options)

        self.grid.fill(Contents.EMPTY)
        total_cells = self.width * self.height

        # pick two distinct linear indices and convert to (x, y)
        point_indexes = self.np_random.choice(total_cells, size=2, replace=False)
        points_of_interest: PositionList = np.divmod(point_indexes, self.height)  # type: ignore
        idx_agent, idx_target = points_of_interest[:2]

        self.grid[idx_agent] = Contents.AGENT
        self.grid[idx_target] = Contents.TARGET
        self.grid[points_of_interest[2:]] = Contents.FAKE_TARGET

        self.obstacle_scheme.generate_obstacles(
            self.grid, points_of_interest, self.np_random
        )

        return self.view(), {}
