"""This module implements a basic container class to easy the dependency injection pattern.
"""

import abc
from dataclasses import dataclass
from typing import MutableMapping, Type, Generic, TypeVar, Callable

Contained = TypeVar("Contained")
ContainedClass = TypeVar("ContainedClass", bound=Type[Contained])


@dataclass(slots=True, frozen=True)
class RegistryInterface(Generic[ContainedClass]):
    interface: abc.ABC
    classes: MutableMapping[str, ContainedClass]

    @abc.abstractmethod
    def register(self, name: str) -> Callable[[ContainedClass], None]:
        """Decorator to register a class definition as part of the Registry."""
