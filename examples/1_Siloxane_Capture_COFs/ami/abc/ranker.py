import abc
from typing import Sequence, Iterator, Optional

from ami.abc.surrogate import Feature, Target
from ami.abc.schema import SchemaProviderInterface
from ami.surrogate_input import SurrogateInput

Index = int


class RankerInterface(SchemaProviderInterface, abc.ABC):

    @abc.abstractmethod
    def rank(self, x: Sequence[Feature]) -> Optional[Iterator[Index]]:
        """Ranks a sequence of feature values from "best" to "worse".

        If 'Iterator[Index]' is returned, 'Index' maps 'x'.
        'None' is returned when no update is necessary.
        """

    @abc.abstractmethod
    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        """Passes known data to the object, always called before self.rank."""
