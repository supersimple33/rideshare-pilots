from abc import ABC, abstractmethod
from typing import SupportsInt

from gymnasium.wrappers import TransformObservation, RecordVideo
import numpy as np

from game.FindCarEnv import FindCarEnv, ObsType
from game.GridGeneration import ObstacleGenerationScheme
from game.utils import Content, Direction, Location, NonNegInt, PosInt
from game.wrappers import car_hider_wrapper, local_view_wrapper
from collections import deque


class HTNFindCar(ABC):
    DEFAULT_N = 7
    DEFAULT_M = 3
    DEFAULT_NUM_FAKE_TARGETS = 0
    DEFAULT_OBSTACLE_SCHEME = None
    DEFAULT_RECORDING_NAME = None

    def __init__(
        self,
        h: PosInt,
        w: PosInt,
        n: PosInt = DEFAULT_N,
        m: PosInt = DEFAULT_M,
        num_fake_targets: NonNegInt = DEFAULT_NUM_FAKE_TARGETS,
        obstacle_scheme: ObstacleGenerationScheme | None = DEFAULT_OBSTACLE_SCHEME,
        recording_name: None | str = DEFAULT_RECORDING_NAME,
    ) -> None:
        env = FindCarEnv(
            width=w,
            height=h,
            num_fake_targets=num_fake_targets,
            render_mode="rgb_array" if recording_name is not None else "human",
            obstacle_scheme=obstacle_scheme,
            check_solvability=True,
        )
        wrapper, obs_space = local_view_wrapper(env.observation_space, n)  # type: ignore
        self._env = TransformObservation(env, wrapper, obs_space)
        wrapper, obs_space = car_hider_wrapper(obs_space, m)
        self._env = TransformObservation(self._env, wrapper, obs_space)
        if recording_name is not None:
            self._env = RecordVideo(
                self._env,
                video_folder="recordings",
                name_prefix=recording_name,
                episode_trigger=lambda _: True,
            )

    @abstractmethod
    def _get_action(
        self,
        observation: ObsType[PosInt, PosInt],
    ) -> Direction:
        pass

    def play_game(self, seed: int | None = None) -> NonNegInt:
        obs, _ = self._env.reset() if seed is None else self._env.reset(seed=seed)
        done = False
        reward = 0.0
        i = 0
        while not done:
            action = self._get_action(obs)  # pyright: ignore[reportArgumentType]
            obs, reward, terminated, truncated, _ = self._env.step(action)
            done = terminated or truncated
            i += 1
        if float(reward) <= 0.0:
            raise RuntimeError("HTN failed while solving the environment.")
        return i

    def close(self) -> None:
        self._env.close()


class DictBasedHTN(HTNFindCar, ABC):
    def __init__(
        self,
        h: PosInt,
        w: PosInt,
        n: PosInt = HTNFindCar.DEFAULT_N,
        m: PosInt = HTNFindCar.DEFAULT_M,
        num_fake_targets: NonNegInt = HTNFindCar.DEFAULT_NUM_FAKE_TARGETS,
        obstacle_scheme: (
            ObstacleGenerationScheme | None
        ) = HTNFindCar.DEFAULT_OBSTACLE_SCHEME,
        recording_name: None | str = HTNFindCar.DEFAULT_RECORDING_NAME,
    ) -> None:
        super().__init__(
            h=h,
            w=w,
            n=n,
            num_fake_targets=num_fake_targets,
            obstacle_scheme=obstacle_scheme,
            recording_name=recording_name,
        )
        self._target_location: None | Location = None
        self._search_path: list[Direction] = []
        self._known_spaces: dict[tuple[SupportsInt, SupportsInt], Content] = {}

    def play_game(self, seed: int | None = None) -> NonNegInt:
        self._target_location = None
        self._search_path = []
        self._known_spaces = {}
        self._visited: set[tuple[SupportsInt, SupportsInt]] = set()
        return super().play_game(seed=seed)

    def _generate_search_path(
        self,
        start: Location,
        target: Location,
    ) -> list[Direction]:
        """Generate a path from start to target using BFS."""
        if np.array_equal(start, target):
            return []

        # BFS storing current nodes and their initial action
        q: deque[tuple[Location, list[Direction]]] = deque(
            (x, y)
            for x, y in [
                (start + Direction.UP, [Direction.UP]),
                (start + Direction.DOWN, [Direction.DOWN]),
                (start + Direction.LEFT, [Direction.LEFT]),
                (start + Direction.RIGHT, [Direction.RIGHT]),
            ]
            if self._known_spaces.get((x[0], x[1]))
            not in [
                Content.OBSTACLE,
                Content.Border,
                Content.OutOfSight,
                Content.FAKE_TARGET,
            ]
        )
        visited: set[tuple[SupportsInt, SupportsInt]] = {(start[0], start[1])}
        while q:
            cur, action_path = q.popleft()
            if np.array_equal(cur, target):
                return action_path
            for action in Direction:
                nxt: Location = cur + action
                if (nxt[0], nxt[1]) in visited:
                    continue
                cell = self._known_spaces.get((nxt[0], nxt[1]))
                if cell is None:
                    # unknown / out of observed area -> treat as blocked
                    continue
                if cell in [
                    Content.OBSTACLE,
                    Content.Border,
                    Content.OutOfSight,
                    Content.FAKE_TARGET,
                ]:
                    # blocked cell
                    continue
                visited.add((nxt[0], nxt[1]))
                q.append((nxt, action_path + [action]))
        raise RuntimeError("No path found to target location.")

    def _get_action(
        self,
        observation: ObsType[PosInt, PosInt],
    ) -> Direction:
        start = observation["agent_position"]
        n = observation["board"].shape[0]
        # self._env.render()
        if self._target_location is not None:
            if not self._search_path:
                self._search_path = self._generate_search_path(
                    start=start,
                    target=self._target_location,
                )
            return self._search_path.pop(0)

        # update observation dictionary
        value: Content
        for (dy, dx), value in np.ndenumerate(  # pyright: ignore[reportAssignmentType]
            observation["board"]
        ):
            y, x = (
                start[0] + dy - n // 2,
                start[1] + dx - n // 2,
            )

            if (
                Content(value) == Content.UNKNOWN_TARGET
                and (y, x) in self._known_spaces
            ):
                continue  # don't overwrite known target with unknown

            if value not in [Content.OutOfSight, Content.Border]:
                self._known_spaces[(y, x)] = (
                    value if value != Content.AGENT else Content.EMPTY
                )

            if Content(value) == Content.TARGET and self._target_location is None:
                self._target_location = np.array([y, x], dtype=np.int32)
                self._search_path = []

        # Use a lookahead check to make sure the search path is still valid otherwise reset it
        if self._search_path:
            next_move = self._search_path[0]
            next_pos = start + next_move
            next_cell = self._known_spaces.get((next_pos[0], next_pos[1]))
            if next_cell not in [Content.EMPTY, Content.TARGET]:
                self._search_path = []

        # grab next action from search path
        self._visited.add((start[0], start[1]))
        if not self._search_path:
            # start a new search to the next unexplored point
            unexplored_points = [
                np.array([y, x], dtype=np.int32)
                for (y, x), value in self._known_spaces.items()
                if value == Content.EMPTY and (y, x) not in self._visited
            ]
            if not unexplored_points:
                raise RuntimeError("No unexplored points left to search.")

            # choose the closest one to the agent (Manhattan distance)
            next_target = min(
                unexplored_points,
                key=lambda p: abs(int(p[0]) - int(start[0]))
                + abs(int(p[1]) - int(start[1])),
            )

            self._search_path = self._generate_search_path(
                start=start,
                target=next_target,
            )

        return self._search_path.pop(0)
