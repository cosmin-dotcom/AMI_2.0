import abc
from typing import Optional, Sequence
from typing import TypeVar

from ami.abc.schema import SchemaProviderInterface

Feature = TypeVar("Feature")
Target = TypeVar("Target")
Index = int


class SurrogateInterface(SchemaProviderInterface, abc.ABC):

    @abc.abstractmethod
    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        """Fit to 'x' descriptor and corresponding 'y' targets."""

    @abc.abstractmethod
    def predict(self, x: Sequence[Feature]) -> Sequence[Target]:
        """Predicts Features based on current model."""
