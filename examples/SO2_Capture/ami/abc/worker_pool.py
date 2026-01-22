import abc
from concurrent.futures import Future
from typing import ContextManager

from ami.abc.factory import FactoryInterface
from ami.abc.worker_factory import WorkerFactoryInterface
from ami.serialized_opaque import SerializedOpaque
from ami.surrogate_input import SurrogateInput

Index = int


class WorkerExecutorInterface(abc.ABC):

    @abc.abstractmethod
    def submit_fit_and_rank(self, inp: SurrogateInput) -> Future:
        """Returns a 'Future' that wraps a 'Optional[Sequence[Index]]'.

        The sequence are indices mapping 'inp.unknown_x' ordered from best to worst.
        If no update is required, the 'Future' wraps 'None'.
        """

    @abc.abstractmethod
    def submit_job(self, inp: SerializedOpaque) -> Future:
        """Returns results from a truth calculation wrapped in a 'Future'."""


class WorkerPoolInterface(ContextManager, abc.ABC):
    pass


class WorkerPoolFactoryInterface(FactoryInterface[WorkerPoolInterface], abc.ABC):

    @abc.abstractmethod
    def set_worker_factory(self, worker_factory: WorkerFactoryInterface) -> None:
        """Sets the worker factory used to spawn workers."""
