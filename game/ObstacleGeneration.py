from abc import ABC, abstractmethod

from game.utils import GridType


class ObstacleGenerationScheme(ABC):
    @abstractmethod
    def generate_obstacles(self, grid: GridType) -> None:
        """Generate obstacles on the grid.

        Args:
            grid: The grid object where obstacles will be placed.
            width: The width of the grid.
            height: The height of the grid.
        """
        pass


class NoObstaclesScheme(ObstacleGenerationScheme):
    def generate_obstacles(self, grid: GridType) -> None:
        """No obstacles are added to the grid."""
        pass
