from enum import Enum, Flag, auto
from typing import Annotated, TypeAlias
from annotated_types import Gt
import numpy as np

GridType: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
PosInt: TypeAlias = Annotated[int, Gt(0)]


class Directions(Enum):
    UP = [np.int32(0), np.int32(1)]
    DOWN = [np.int32(0), np.int32(-1)]
    RIGHT = [np.int32(1), np.int32(0)]
    LEFT = [np.int32(-1), np.int32(0)]


class Contents(np.uint8, Flag):
    EMPTY = auto()
    OBSTACLE = auto()
    TARGET = auto()
    FAKE_TARGET = auto()
    AGENT = auto()
