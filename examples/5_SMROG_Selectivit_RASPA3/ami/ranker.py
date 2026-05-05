from dataclasses import dataclass
from typing import Sequence, Iterator, Callable

import ami.abc
from ami.abc import SchemaInterface, Feature, Target
from ami.abc.ranker import Index


@dataclass(frozen=True, slots=True)
class SingleSurrogateRanker(ami.abc.RankerInterface):
    surrogate: ami.abc.SurrogateInterface
    acquisition_function: Callable[[Iterator[Target]], Iterator[Index]]

    def rank(self, x: Sequence[Feature]) -> Iterator[Index]:
        y = self.surrogate.predict(x)
        return self.acquisition_function(y)

    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        self.surrogate.fit(x, y)

    def schema(self) -> SchemaInterface:
        return self.surrogate.schema()
