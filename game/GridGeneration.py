from abc import ABC, abstractmethod

import numpy as np

from game.utils import CARDINAL_DIRECTIONS, Contents, GridType, PositionList


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


