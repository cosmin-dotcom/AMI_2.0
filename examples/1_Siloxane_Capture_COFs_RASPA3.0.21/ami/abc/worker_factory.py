import abc

from ami.abc.calculator import CalculatorInterface
from ami.abc.factory import FactoryInterface
from ami.abc.ranker import RankerInterface
from ami.abc.worker import WorkerInterface


class WorkerFactoryInterface(FactoryInterface[WorkerInterface], abc.ABC):

    @abc.abstractmethod
    def set_ranker(self, ranker: RankerInterface) -> None:
        """Sets the underlying surrogate model to 'surrogate'."""

    @abc.abstractmethod
    def set_truth(self, truth: CalculatorInterface) -> None:
        """Sets the underlying truth source to 'truth'."""
