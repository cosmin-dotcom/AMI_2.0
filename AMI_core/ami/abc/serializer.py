import abc
from typing import Generic

from ami.serialized_opaque import SerializedOpaque, T


class Serializer(Generic[T], abc.ABC):
    """Transforms data to and from bytes for persistence and transfer."""

    @classmethod
    @abc.abstractmethod
    def encode(cls, obj: T) -> SerializedOpaque[T]:
        """Encodes obj of original type T into opaque bytes."""
