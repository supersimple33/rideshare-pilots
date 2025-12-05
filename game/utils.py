import itertools
from enum import Enum, auto, unique
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar, Iterator, Optional
from collections import deque

from annotated_types import Gt, Ge
import numpy as np
from gymnasium.spaces import Space

GridType: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
PosInt: TypeAlias = Annotated[int, Gt(0)]
NonNegInt: TypeAlias = Annotated[int, Ge(0)]
Location: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
PositionList: TypeAlias = list[Location]

T = TypeVar("T")
H = TypeVar("H", bound=PosInt)
W = TypeVar("W", bound=PosInt)
K = TypeVar("K", bound=PosInt)
DT = TypeVar("DT", bound=np.generic)


@unique
class Direction(tuple[np.int32, np.int32], Enum):
    DOWN = (np.int32(1), np.int32(0))
    UP = (np.int32(-1), np.int32(0))
    RIGHT = (np.int32(0), np.int32(1))
    LEFT = (np.int32(0), np.int32(-1))


@unique
class Content(np.uint8, Enum):
    EMPTY = 0
    OBSTACLE = auto()
    TARGET = auto()
    FAKE_TARGET = auto()
    UNKNOWN_TARGET = auto()
    AGENT = auto()
    Border = auto()
    OutOfSight = auto()


def check_all_adj_empty(grid: GridType, y: np.integer, x: np.integer) -> np.bool_:
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
            max(0, y - 1) : min(grid.shape[0], y + 2),
            max(0, x - 1) : min(grid.shape[1], x + 2),
        ]
        == Content.EMPTY
    )


def window_at(
    a: np.ndarray[tuple[PosInt, PosInt], np.dtype[DT]],
    loc: Location,
    k: K,
    pad_value: DT,
) -> np.ndarray[tuple[K, K], np.dtype[DT]]:
    """Return a k×k window centered at (i, j) with zero-padding as needed.

    The return array is annotated to have the same dtype as the input array.
    """
    assert k > 0, "k must be positive"
    i, j = loc
    n, m = a.shape
    half = k // 2

    # Compute actual slice bounds (clamped to valid region)
    i0 = max(i - half, 0)
    i1 = min(i + half + 1, n)
    j0 = max(j - half, 0)
    j1 = min(j + half + 1, m)

    # Extract visible sub-array
    view = a[i0:i1, j0:j1]

    # Fast path: fully inside boundaries
    if view.shape == (k, k):
        return view

    # Otherwise, pad manually into k×k output
    out: np.ndarray[tuple[K, K], np.dtype[DT]]
    out = np.full(
        (k, k), pad_value, dtype=a.dtype
    )  # pyright: ignore[reportAssignmentType]

    # Compute placement inside padded output
    top = half - (i - i0)
    left = half - (j - j0)
    out[top : top + view.shape[0], left : left + view.shape[1]] = view
    return out


def reachable_window_at(
    a: np.ndarray[tuple[PosInt, PosInt], np.dtype[DT]],
    loc: Location,
    k: K,
    pad_value: DT,
) -> np.ndarray[tuple[K, K], np.dtype[DT]]:
    """Return a k×k window centered at (i, j) with padding for unreachable cells.

    Unreachable cells are those that are obstacles in the grid.
    The return array is annotated to have the same dtype as the input array.
    """

    # Needs a copy to modifications do not affect original array
    standard_window = window_at(a, loc, k, pad_value).copy()

    cx = k // 2
    cy = k // 2

    def is_walkable(cell: DT) -> bool:
        return cell not in [Content.OBSTACLE, Content.Border]

    reachable = np.zeros((k, k), dtype=bool)

    q: deque[tuple[np.integer | int, np.integer | int]] = deque()
    q.append((cy, cx))
    reachable[cy, cx] = True

    while q:
        y, x = q.popleft()
        for dy, dx in Direction:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < k
                and 0 <= nx < k
                and not reachable[ny, nx]
                and is_walkable(standard_window[ny, nx])  # type: ignore
            ):
                reachable[ny, nx] = True
                q.append((ny, nx))

    visible = reachable.copy()
    for y in range(k):
        for x in range(k):
            if reachable[y, x]:
                for dy, dx in Direction:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < k and 0 <= nx < k:
                        visible[ny, nx] = True
    for y in range(k):
        for x in range(k):
            if not visible[y, x] and standard_window[y, x] != pad_value:
                standard_window[y, x] = Content.OutOfSight  # type: ignore
    return standard_window


def obscure_cars(
    board: np.ndarray[tuple[K, K], np.dtype[DT]],
    n: PosInt,
    copy: bool = True,
) -> np.ndarray[tuple[K, K], np.dtype[DT]]:
    """Obscure all cars in the board except those within n distance (manhattan) from the center.

    Args:
        board: The board to obscure.
        n: The distance from the center to keep cars visible.
    Returns:
        The obscured board.
    """
    k = board.shape[0]
    center = (k // 2, k // 2)
    obscured_board = board.copy() if copy else board
    for y, x in itertools.product(range(k), range(k)):
        if (
            board[y, x] in [Content.TARGET, Content.FAKE_TARGET]
            and abs(y - center[0]) + abs(x - center[1]) > n
        ):
            obscured_board[y, x] = Content.UNKNOWN_TARGET
    return obscured_board


def content_to_one_hot(
    board: np.ndarray[tuple[H, W], np.dtype[np.uint8]],
) -> np.ndarray[tuple[H, W, PosInt], np.dtype[np.uint8]]:
    """Convert a board to a one-hot encoded representation.

    Args:
        board: The board to convert.
        num_content_types: The number of different content types in the grid.

    Returns:
        A one-hot encoded representation of the board.
    """
    return np.eye(len(Content), dtype=np.uint8)[board]


class RandomSet(Generic[T]):
    """A set supporting O(1) add, remove, contains, and random-pop using a NumPy RNG."""

    def __init__(
        self,
        iterable: Optional[Iterator[T]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.data: dict[T, int] = {}
        self.items: list[T] = []
        self.rng: np.random.Generator = (
            np.random.default_rng(42) if rng is None else rng
        )
        if iterable is not None:
            for val in iterable:
                self.add(val)

    def add(self, val: T) -> None:
        if val not in self.data:
            self.data[val] = len(self.items)
            self.items.append(val)

    def remove(self, val: T) -> None:
        """Remove val, raising KeyError if not present."""
        if val not in self.data:
            raise KeyError(val)
        idx = self.data.pop(val)
        last = self.items.pop()
        if idx < len(self.items):
            self.items[idx] = last
            self.data[last] = idx

    def discard(self, val: T) -> None:
        """Remove val if present; do nothing otherwise."""
        if val in self.data:
            self.remove(val)

    def pop_random(self) -> T:
        """Pop and return a random element in O(1)."""
        if not self.items:
            raise KeyError("pop from empty set")
        idx = int(self.rng.integers(len(self.items)))
        val = self.items[idx]
        self.remove(val)
        return val

    def random(self) -> T:
        """Return a random element without removing it."""
        if not self.items:
            raise KeyError("sample from empty set")
        idx = int(self.rng.integers(len(self.items)))
        return self.items[idx]

    def __contains__(self, val: object) -> bool:
        return val in self.data

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __repr__(self) -> str:
        return f"RandomSet({{{', '.join(map(repr, self.items))}}})"


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
        mask: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None,
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
