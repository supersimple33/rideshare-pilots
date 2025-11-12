from typing import Any, Generic, Literal, TypeAlias, TypedDict, cast

import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace, MultiDiscrete
import numpy as np
from w9_pathfinding.envs import Grid  # type: ignore
from w9_pathfinding.pf import BiAStar  # type: ignore

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

    def __annotate__(self) -> dict:  # TODO: fix these annotation errors with generics
        """A TypedDict representing the observation type of the FindCarEnv."""
        return {}


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
        obstacle_scheme: ObstacleGenerationScheme | None = None,
        render_mode: str | None = None,
        check_solvability: bool = True,
    ):
        """Initialize the FindCar environment.

        Args:
            width: The width of the grid.
            height: The height of the grid.
            obstacle_scheme: The obstacle generation scheme to use when generating the grid. Defaults to no obstacles.
        """
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self.obstacle_scheme = obstacle_scheme or NOOP_GENERATION_SCHEME
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.num_fake_targets = num_fake_targets
        self.check_solvability = check_solvability

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
            (height, width), dtype=np.uint8
        )  # TODO: Fix the complaints here

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
        points_of_interest = self._generate_agent_and_targets()

        self.obstacle_scheme.generate_obstacles(
            self.grid,
            points_of_interest,
            self.np_random,
        )

        while not self._check_solvability() and self.check_solvability:
            self.grid.fill(Content.EMPTY)
            points_of_interest = self._generate_agent_and_targets()

            self.obstacle_scheme.generate_obstacles(
                self.grid,
                points_of_interest,
                self.np_random,
            )

        return self.view(), {}

    def _generate_agent_and_targets(self) -> list[Location]:
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
            y, x = self._target_location
            if i >= MAX_PLACEMENT_ATTEMPTS:
                raise RuntimeError("Failed to place target after maximum attempts")
            if not check_adj_empty(self.grid, y, x):
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
                y, x = fake_target_location
                if i >= MAX_PLACEMENT_ATTEMPTS:
                    raise RuntimeError("Failed to place target after maximum attempts")
                if not check_adj_empty(self.grid, y, x):
                    i += 1
                    continue
                break
            self.grid[tuple(fake_target_location)] = Content.FAKE_TARGET
            points_of_interest.append(fake_target_location)
        return points_of_interest

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

    def _check_solvability(self) -> bool:
        """Check if there is a valid path from the agent to the target using A*.

        Returns:
            True if a path exists, False otherwise.
        """
        transformed_grid = np.where(
            (self.grid == Content.EMPTY)
            | (self.grid == Content.TARGET)
            | (self.grid == Content.AGENT),
            1,
            -1,
        )
        grid_env = Grid(transformed_grid)  # pyright: ignore[reportUnknownVariableType]
        y1, x1 = self._agent_location
        y2, x2 = self._target_location
        path = cast(
            list[tuple[int, int]] | None,
            BiAStar(grid_env).find_path(  # pyright: ignore[reportUnknownMemberType]
                (x1, y1), (x2, y2)
            ),
        )
        return bool(path)

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

    def _render_rgb_array(
        self,
    ) -> np.ndarray[tuple[H, W, Literal[3]], np.dtype[np.uint8]]:
        """Render the current state of the environment to an RGB array.

        Returns:
            An RGB array representing the current state of the environment.
        """
        # Define colors for each content type (R, G, B)
        color_map = {
            Content.EMPTY.value: np.array([255, 255, 255], dtype=np.uint8),  # white
            Content.OBSTACLE.value: np.array([30, 30, 30], dtype=np.uint8),  # gray
            Content.TARGET.value: np.array([0, 200, 0], dtype=np.uint8),  # green
            Content.FAKE_TARGET.value: np.array(
                [0, 100, 255], dtype=np.uint8
            ),  # blue-ish
            Content.AGENT.value: np.array([200, 0, 0], dtype=np.uint8),  # red
        }

        img: np.ndarray[tuple[H, W, Literal[3]], np.dtype[np.uint8]]
        img = np.zeros(
            (self.height, self.width, 3), dtype=np.uint8
        )  # pyright: ignore[reportAssignmentType]

        # Map grid values to colors
        for y in range(self.height):
            for x in range(self.width):
                val = int(self.grid[y, x])
                img[y, x] = color_map.get(
                    val, np.array([128, 128, 128], dtype=np.uint8)
                )

        return img

    def render(self) -> None | np.ndarray[tuple[H, W, Literal[3]], np.dtype[np.uint8]]:
        """Render the environment."""
        match self.render_mode:
            case "human":
                self._render_human()
            case "rgb_array":
                return self._render_rgb_array()
            case None:
                pass
            case _:
                raise NotImplementedError(
                    f"Render mode {self.render_mode} is not implemented."
                )
