from collections.abc import Mapping
from typing import Generic, TypeVar
from typing import Protocol

T = TypeVar("T")


class SerializedOpaque(Mapping[str, bytes], Generic[T]):
    pass
