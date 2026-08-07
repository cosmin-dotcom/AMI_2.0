import abc

from ami.option import Option, T, Generic

Server = Client = int


class SendInterface(Generic[T], abc.ABC):
    """Defines a transport interface between a ami.scheduler.SchedulerInterface and a ami.worker.WorkerInterface."""

    @abc.abstractmethod
    def send(self) -> Option[T]:
        """Returns a job descriptor used to spawn an actual job."""


class ReceiveInterface(Generic[T], abc.ABC):
    """Defines a transport interface between a ami.scheduler.SchedulerInterface and a ami.worker.WorkerInterface."""

    @abc.abstractmethod
    def recv(self, msg: Option[T]) -> None:
        """Returns a job descriptor used to spawn an actual job."""


class TransportInterface(Generic[T], abc.ABC):

    def serve(self) -> Server:
        """Sets up the end-point."""

    def connect(self) -> Client:
        """Connect to the end-point."""
