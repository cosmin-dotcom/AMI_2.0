import abc
from typing import TypeVar, Sequence, Tuple
from ami.serialized_opaque import SerializedOpaque
from ami.option import Option
from ami.surrogate_input import SurrogateInput

ResultsType = TypeVar("ResultsType")
Index = int

class SchedulerInterface(abc.ABC):
    """Defines a job Scheduler interface which maintains a prioritised list of jobs.
    """

    @abc.abstractmethod
    def set_result(self, index: Index, value: Option[SerializedOpaque]):
        """Sets truth results for 'index'. If 'Nothing', sets the calculation as 'failed'."""

    @abc.abstractmethod
    def set_ranks(self, ranks: Option[Sequence[Index]]):
        """Sets ranking of internal indices. If 'Nothing', assume there is not change."""

    @abc.abstractmethod
    def needs_new_ranking(self) -> bool:
        """'True' if the scheduler wants new rankings to be estimated, else 'False'."""

    @abc.abstractmethod
    def ranker_inputs(self) -> Tuple[Sequence[Index], SurrogateInput]:
        """Returns a sequence of remaining unkown indices with matching 'SurrogateInput' structure.

        In particular,

            >>> seq, surr_input = self.ranker_inputs()
            >>> len(seq) == len(surr_input.unknown_x)
            True
        """

    @abc.abstractmethod
    def next(self) -> Index:
        """Returns the next "best" index using current surrogate rankings."""

    @abc.abstractmethod
    def parameters(self, index: Index) -> SerializedOpaque:
        """Returns parameters required for truth calculations at 'index'."""
