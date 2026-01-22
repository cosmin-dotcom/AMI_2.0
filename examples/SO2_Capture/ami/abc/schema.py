import abc
from typing import Type, Tuple, Iterator


class SchemaInterface(abc.ABC):
    @abc.abstractmethod
    def feature(self) -> Iterator[Tuple[str, Type]]:
        """Series of (name, type), in order, representing a Feature."""

    @abc.abstractmethod
    def target(self) -> Iterator[Tuple[str, Type]]:
        """Series of (name, type), in order, representing a Target."""


class SchemaProviderInterface(abc.ABC):

    @abc.abstractmethod
    def schema(self) -> SchemaInterface:
        """Returns a pair of sequences of (name, type) which represents Feature and Target formats."""
