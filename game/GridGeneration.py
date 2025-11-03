from abc import ABC, abstractmethod

import numpy as np

from game.utils import Content, Direction, GridType, PositionList, check_adj_empty


class ObstacleGenerationScheme(ABC):

    @abstractmethod
    def generate_obstacles(
        self,
        grid: GridType,
        points_of_interest: PositionList,
        generator: np.random.Generator,
    ) -> None:
        """Generate obstacles on the grid.

        Args:
            grid: The grid object where obstacles will be placed.
            width: The width of the grid.
            height: The height of the grid.
        """
        pass


class NoObstaclesScheme(ObstacleGenerationScheme):
    def generate_obstacles(
        self,
        grid: GridType,
        points_of_interest: PositionList,
        generator: np.random.Generator,
    ) -> None:
        """No obstacles are added to the grid."""
        pass


class DisjointBlobs(ObstacleGenerationScheme):
    def __init__(self, blob_count: int, blob_size: int) -> None:
        """Initialize the DisjointBlobs obstacle generation scheme.
        Args:
            blob_count: The number of disjoint blobs to generate.
            blob_size: The size of each blob.
        """
        self.blob_count = blob_count
        self.blob_size = blob_size

    def generate_obstacles(
        self,
        grid: GridType,
        points_of_interest: PositionList,
        generator: np.random.Generator,
    ) -> None:
        """Generate blobs which are non-adjacent on the grid.

        Args:
            grid: The grid object where obstacles will be placed.
            points_of_interest: unused.
            generator: Random number generator to use for obstacle placement.
        """
        num_cells = grid.size

        attempts = 0
        max_attempts = self.blob_count * 10
        blobs_created = 0
        # Try to create the specified number of blobs stopping after a maximum number of attempts
        while blobs_created < self.blob_count and attempts < max_attempts:
            # check that we can place a blob starting at a random position
            attempts += 1
            start_idx = generator.integers(0, num_cells)
            x, y = np.divmod(start_idx, grid.shape[1])
            if check_adj_empty(grid, x, y):
                continue

            # setup tracking for the new blob
            blob_area = [(x, y)]
            blob_checked = set(blob_area)
            blob_frontier: list[tuple[np.integer, np.integer]] = [
                (x + x_nudge, y + y_nudge)
                for x_nudge, y_nudge in Direction
                if 0 <= x + x_nudge < grid.shape[0] and 0 <= y + y_nudge < grid.shape[1]
            ]
            blob_size = 1

            # expand the blob until it reaches the desired size
            while blob_frontier and blob_size < self.blob_size:
                cx, cy = blob_frontier.pop(generator.integers(0, len(blob_frontier)))
                blob_checked.add((cx, cy))

                # ensure the new cell is valid
                if check_adj_empty(grid, cx, cy):
                    continue

                # add the new cell to the blob
                blob_area.append((cx, cy))
                blob_size += 1

                # expand the frontier
                for x_nudge, y_nudge in Direction:
                    nx, ny = cx + x_nudge, cy + y_nudge
                    if (
                        0 <= nx < grid.shape[0]
                        and 0 <= ny < grid.shape[1]
                        and (nx, ny) not in blob_checked
                    ):
                        blob_frontier.append((nx, ny))

            # when we are done growing the blob, mark it on the grid
            for bx, by in blob_area:
                grid[bx, by] = Content.OBSTACLE
