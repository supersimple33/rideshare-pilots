from enum import Enum, auto
from typing import Annotated, Literal, TypeAlias

from annotated_types import Gt, Ge
import numpy as np

GridType: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
PositionList: TypeAlias = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int32]]
PosInt: TypeAlias = Annotated[int, Gt(0)]
NonNegInt: TypeAlias = Annotated[int, Ge(0)]
Location: TypeAlias = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]


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
