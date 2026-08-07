import abc
from typing import TypeVar

from ami.abc.schema import SchemaProviderInterface
from ami.option import Generic
from ami.serialized_opaque import SerializedOpaque

OpaqueParameters = TypeVar("OpaqueParameters", bound=SerializedOpaque)
OpaqueResults = TypeVar("OpaqueResults", bound=SerializedOpaque)
Index = int


class CalculatorInterface(Generic[OpaqueParameters, OpaqueResults], SchemaProviderInterface, abc.ABC):
    """Describes a calculator runnable by a ami.worker.Worker process.
    """

    @abc.abstractmethod
    def calculate(self, inp: OpaqueParameters) -> OpaqueResults:
        """Returns results from a truth calculation."""
