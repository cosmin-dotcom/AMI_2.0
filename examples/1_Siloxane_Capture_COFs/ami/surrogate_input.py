from dataclasses import dataclass
from typing import Sequence, Any


@dataclass(frozen=True, slots=True)
class SurrogateInput:
    known_x: Sequence[Any]
    known_y: Sequence[Any]
    unknown_x: Sequence[Any]
