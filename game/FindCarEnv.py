from typing import Any, Generic, TypeAlias, TypedDict

import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace, MultiDiscrete
import numpy as np

from game.GridGeneration import NoObstaclesScheme, ObstacleGenerationScheme
from game.utils import (
    Content,
    Direction,
    FiniteSet,
    Location,
    NonNegInt,
    H,
    W,
    check_adj_empty,
)

NOOP_GENERATION_SCHEME = NoObstaclesScheme()
MAX_PLACEMENT_ATTEMPTS = 1000


class ObsType(TypedDict, Generic[H, W]):
    agent_position: Location
    board: np.ndarray[tuple[H, W], np.dtype[np.uint8]]


ActionType: TypeAlias = Direction


class FindCarEnv(gym.Env[ObsType[H, W], ActionType], Generic[H, W]):
    """A mini grid environment. It can generate a different layout each time according to
    the generation method specified. The objective is to move an agent to the correct car
    which will all look identical to the agent until it is very close to it."""

    width: W
    height: H
    obstacle_scheme: ObstacleGenerationScheme
    grid: np.ndarray[tuple[H, W], np.dtype[np.uint8]]
    _target_location: Location
    _agent_location: Location

    def __init__(
        self,
        width: W,
        height: H,
        num_fake_targets: NonNegInt = 0,
        obstacle_scheme: ObstacleGenerationScheme = NOOP_GENERATION_SCHEME,
        render_mode: str | None = None,
    ):
        """Initialize the FindCar environment.

        Args:
            width: The width of the grid.
            height: The height of the grid.
            obstacle_scheme: The obstacle generation scheme to use when generating the grid. Defaults to no obstacles.
        """
        self.metadata = {"render_modes": ["console"], "render_fps": 4}
        self.obstacle_scheme = obstacle_scheme
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.num_fake_targets = num_fake_targets

        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = DictSpace(
            {
                "agent_position": MultiDiscrete([height, width], dtype=np.int32),
                "board": MultiDiscrete(
                    np.full((height, width), len(Content)),
                    dtype=np.uint8,
                ),
            }
        )  # pyright: ignore[reportAttributeAccessIssue]
        self.action_space = FiniteSet(
            [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        )

        self.grid = np.empty(
            (height, width), dtype=np.int8
        )  # pyright: ignore[reportAttributeAccessIssue]

    def view(self) -> ObsType[H, W]:
        """Get the current observation of the environment.

        Returns:
            The current observation as a dictionary.
        """
        return {
            "agent_position": self._agent_location,
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

        self.grid.fill(Content.EMPTY)

        # generate a random position for the agent
        self._agent_location = self.np_random.integers(
            low=0, high=[self.height, self.width], size=(2,), dtype=np.int32
        )
        self.grid[tuple(self._agent_location)] = Content.AGENT
        # generate a random position for the target and ensure it is not adjacent to anything else
        i = 0
        while True:
            self._target_location = self.np_random.integers(
                low=0, high=[self.height, self.width], size=(2,), dtype=np.int32
            )
            x, y = self._target_location
            if i >= MAX_PLACEMENT_ATTEMPTS:
                raise RuntimeError("Failed to place target after maximum attempts")
            if not check_adj_empty(self.grid, x, y):
                i += 1
                continue
            break
        self.grid[tuple(self._target_location)] = Content.TARGET
        # generate fake target locations
        points_of_interest = [self._agent_location, self._target_location]
        for _ in range(self.num_fake_targets):
            while True:
                fake_target_location = self.np_random.integers(
                    low=0, high=[self.height, self.width], size=(2,), dtype=np.int32
                )
                x, y = fake_target_location
                if i >= MAX_PLACEMENT_ATTEMPTS:
                    raise RuntimeError("Failed to place target after maximum attempts")
                if not check_adj_empty(self.grid, x, y):
                    i += 1
                    continue
                break
            self.grid[tuple(fake_target_location)] = Content.FAKE_TARGET
            points_of_interest.append(fake_target_location)

        self.obstacle_scheme.generate_obstacles(
            self.grid,
            points_of_interest,
            self.np_random,
        )

        return self.view(), {}

    def step(
        self, action: ActionType
    ) -> tuple[ObsType[H, W], float, bool, bool, dict[str, str]]:
        """Take a step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the new observation, reward, done flag, truncated flag, and info dictionary.
        """
        self.grid[tuple(self._agent_location)] = Content.EMPTY
        self._agent_location += action

        # check if the new position is valid
        y, x = self._agent_location
        if not (0 <= x < self.width and 0 <= y < self.height):
            return self.view(), -1.0, True, False, {"reason": "out_of_bounds"}

        if self.grid[y, x] == Content.OBSTACLE:
            return self.view(), -1.0, True, False, {"reason": "hit_obstacle"}
        if self.grid[y, x] == Content.TARGET:
            return self.view(), 1.0, True, False, {"reason": "found_target"}
        if self.grid[y, x] == Content.FAKE_TARGET:
            return self.view(), -0.5, True, False, {"reason": "found_fake_target"}
        self.grid[y, x] = Content.AGENT
        return self.view(), 0.0, False, False, {}

    def _render_human(self) -> None:
        """Render the current state of the environment to the console."""
        content_symbols = {
            Content.EMPTY: ".",
            Content.OBSTACLE: "#",
            Content.TARGET: "T",
            Content.FAKE_TARGET: "F",
            Content.AGENT: "A",
        }
        for y in range(self.height):
            row = "".join(
                content_symbols.get(self.grid[y, x], "?") + " "
                for x in range(self.width)
            )
            print(row)
        print()

    def render(self) -> None:
        """Render the environment."""
        match self.render_mode:
            case "human":
                self._render_human()
            case None:
                pass
            case _:
                raise NotImplementedError(
                    f"Render mode {self.render_mode} is not implemented."
                )
