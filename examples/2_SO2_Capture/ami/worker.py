from dataclasses import dataclass
from typing import Sequence, Iterator, Optional

import ami.abc
from ami.abc import SchemaInterface, Feature, Target, CalculatorInterface, RankerInterface
from ami.abc.ranker import Index
from ami.factory import DataclassFactory
from ami.serialized_opaque import SerializedOpaque


@dataclass(frozen=True, slots=True)
class SharedMemorySingleThreadWorker(ami.abc.WorkerInterface):
    truth: ami.abc.CalculatorInterface
    ranker: ami.abc.RankerInterface

    def calculate(self, inp: SerializedOpaque) -> SerializedOpaque:
        return self.truth.calculate(inp)

    def rank(self, x: Sequence[Feature]) -> Optional[Iterator[Index]]:
        return self.ranker.rank(x)

    def fit(self, x: Sequence[Feature], y: Sequence[Target]) -> None:
        return self.ranker.fit(x, y)

    def schema(self) -> SchemaInterface:
        pass


@dataclass(frozen=True, slots=True)
class ShareMemorySingleThreadWorkerFactory(DataclassFactory, ami.abc.WorkerFactoryInterface):
    dataclass = SharedMemorySingleThreadWorker

    def set_ranker(self, ranker: RankerInterface) -> None:
        self.set("ranker", ranker)

    def set_truth(self, truth: CalculatorInterface) -> None:
        self.set("truth", truth)
