from enum import Enum, auto
from typing import Annotated, Literal, TypeAlias

from annotated_types import Gt, Ge
import numpy as np

GridType: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
PosInt: TypeAlias = Annotated[int, Gt(0)]
NonNegInt: TypeAlias = Annotated[int, Ge(0)]
Location: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
PositionList: TypeAlias = list[Location]


class Direction(tuple[np.int32, np.int32], Enum):
    UP = (np.int32(0), np.int32(1))
    DOWN = (np.int32(0), np.int32(-1))
    RIGHT = (np.int32(1), np.int32(0))
    LEFT = (np.int32(-1), np.int32(0))


class Content(np.uint8, Enum):
    EMPTY = auto()
    OBSTACLE = auto()
    TARGET = auto()
    FAKE_TARGET = auto()
    AGENT = auto()


def check_adj_empty(grid: GridType, x: np.integer, y: np.integer) -> np.bool:
    """Check that the cell at (x, y) and its adjacent cells are empty.

    Args:
        grid: The grid to check.
        x: The x coordinate of the cell.
        y: The y coordinate of the cell.
    Returns:
        True if the cell and its adjacent cells are empty, False otherwise.
    """
    return np.all(
        grid[
            max(0, x - 1) : min(grid.shape[0], x + 2),
            max(0, y - 1) : min(grid.shape[1], y + 2),
        ]
        == Content.EMPTY
    )
