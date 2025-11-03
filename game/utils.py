from enum import Enum, auto
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar

from annotated_types import Gt, Ge
import numpy as np
from gymnasium.spaces import Space

GridType: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
PosInt: TypeAlias = Annotated[int, Gt(0)]
NonNegInt: TypeAlias = Annotated[int, Ge(0)]
Location: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
PositionList: TypeAlias = list[Location]


class Direction(tuple[np.int32, np.int32], Enum):
    DOWN = (np.int32(0), np.int32(1))
    UP = (np.int32(0), np.int32(-1))
    RIGHT = (np.int32(1), np.int32(0))
    LEFT = (np.int32(-1), np.int32(0))


class Content(np.uint8, Enum):
    EMPTY = auto()
    OBSTACLE = auto()
    TARGET = auto()
    FAKE_TARGET = auto()
    AGENT = auto()


T = TypeVar("T")


class FiniteSet(Space[T], Generic[T]):
    """A Gym space representing a finite set of allowed (immutable) actions."""

    n: PosInt
    elements: list[T]

    def __init__(self, elements: list[T], seed: int | None = None):
        # elements: shape (N, d); dtype is enforced for actions
        assert elements, "FiniteSet must have at least one element."
        self.elements = elements
        super().__init__(
            shape=elements[0].shape if hasattr(elements[0], "shape") else (1,),  # type: ignore
            dtype=type(elements[0]),
            seed=seed,
        )

    def sample(
        self,
        mask: np.ndarray[tuple[int], np.dtype[np.int8]] | None = None,
        probability: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    ) -> T:
        if mask is not None and probability is not None:
            raise ValueError("Cannot use both mask and probability for sampling.")

        chosen_index: int | np.int64
        if mask is not None:
            valid_indices = np.nonzero(mask)[0]
            chosen_index = self.np_random.choice(valid_indices)
            return self.elements[chosen_index]
        if probability is not None:
            if np.sum(probability) != 1.0:
                raise ValueError("Probability distribution must sum to 1.")
            probability = probability / np.sum(probability)
            chosen_index = self.np_random.choice(self.n, p=probability)
            return self.elements[chosen_index]

        chosen_index = self.np_random.integers(0, len(self.elements))
        return self.elements[chosen_index]

    def contains(self, x: T) -> bool:
        return x in self.elements


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
