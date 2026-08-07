import abc

from ami.abc.data_manager import DataManagerInterface
from ami.abc.factory import FactoryInterface
from ami.abc.ranker import RankerInterface
from ami.abc.scheduler import SchedulerInterface
from ami.abc.schema import SchemaInterface
from ami.abc.worker_pool import WorkerPoolInterface


class SchedulerFactoryInterface(FactoryInterface[SchedulerInterface], abc.ABC):
    """Defines a 'Scheduler' factory/builder"""

    @abc.abstractmethod
    def set_data_manager(self, data_manager: DataManagerInterface):
        """Sets data manager."""

    @abc.abstractmethod
    def set_ranker_schema(self, schema: SchemaInterface) -> None:
        """Sets the typing schema used by the surrogate model."""

    @abc.abstractmethod
    def set_truth_schema(self, schema: SchemaInterface) -> None:
        """Sets the typing schema used by the truth source."""

    @abc.abstractmethod
    def set_worker_pool(self, worker_pool: WorkerPoolInterface) -> None:
        """Sets the endpoint to listen to for communication with workers."""

    @abc.abstractmethod
    def set_initial_ranker(self, ranker: RankerInterface) -> None:
        """Sets ranker to kickstart order in the scheduler."""
