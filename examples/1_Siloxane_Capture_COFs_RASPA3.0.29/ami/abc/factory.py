import abc
from typing import Iterator, TypeVar, Generic

from ami.option import Option

BuiltObject = TypeVar("BuildObject")


class FactoryInterface(Generic[BuiltObject], abc.ABC):

    @abc.abstractmethod
    def build(self) -> Option[BuiltObject]:
        """Returns Some(object) if factory is finalized else Nothing."""

    @classmethod
    @abc.abstractmethod
    def _fields(cls) -> Iterator[str]:
        """Returns an iterator of field names."""

    @abc.abstractmethod
    def _is_finalized(self) -> bool:
        """Returns 'True' if the object can be build, 'False' otherwise."""
