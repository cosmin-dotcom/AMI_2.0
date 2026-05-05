import abc
from typing import Sequence, Collection, Tuple

from ami.abc.calculator import OpaqueParameters
from ami.abc.surrogate import Feature, Target
from ami.option import Option
from ami.result import Result

Index = int


class StateMachineInterface(abc.ABC):
    """State machine to keep track of known targets, unknown ones and what is running.

    State machine
    -------------

    1. self.reset and default value is (not done, available, not failed)
    2. self.select transforms state to (not done, not available, not failed)
    3. self.set transforms state to (done, not available, failed/not failed)
    """

    @abc.abstractmethod
    def select(self, index: Index) -> None:
        """(not done, available, not failed) -> (not done, not available, not failed)"""

    @abc.abstractmethod
    def set(self, index: Index, success: bool) -> None:
        """Sets value, (not done, not available, not failed) -> (done, not available, 'success')"""

    @abc.abstractmethod
    def reset(self, index: Index) -> None:
        """(*, *, *) -> (not done, available, not failed)"""

    @abc.abstractmethod
    def list_done(self, include_failures=False) -> Collection[bool]:
        """Returns list of bools, 'True' if calculation done, else 'False'."""

    @abc.abstractmethod
    def list_available(self) -> Collection[bool]:
        """Returns list of bools, 'True' if not done and not running, else 'False'."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples managed by the state machine."""


class SurrogateProviderInterface(abc.ABC):
    @abc.abstractmethod
    def unknown(self, state: StateMachineInterface) -> Sequence[Feature]:
        """Returns sequences of all features for whom not calculation has been finished nor scheduled. Immutable."""

    @abc.abstractmethod
    def known(self, state: StateMachineInterface) -> Tuple[Sequence[Feature], Sequence[Target]]:
        """Returns (features, targets) for all successfully finished calculations. Both immutable."""

    @abc.abstractmethod
    def set_target(self, index: Index, value: Option[Target]) -> None:
        """Sets internal target value to 'value' if Some(value). Ignore otherwise."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples managed by the surrogate provider."""


class StatelessSurrogateProviderInterface(abc.ABC):
    @abc.abstractmethod
    def unknown(self) -> Sequence[Feature]:
        """Returns sequences of all features for whom not calculation has been finished nor scheduled. Immutable."""

    @abc.abstractmethod
    def known(self) -> Tuple[Sequence[Feature], Sequence[Target]]:
        """Returns (features, targets) for all successfully finished calculations. Both immutable."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples managed by the surrogate provider."""


class TruthProviderInterface(abc.ABC):
    @abc.abstractmethod
    def parameters(self, index: Index, state: StateMachineInterface) -> Option[OpaqueParameters]:
        """Returns parameters at index 'index'. They must be bytes-like and will be decoded by the calculator."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples managed by the truth provider."""


class StatelessTruthProviderInterface(abc.ABC):
    @abc.abstractmethod
    def parameters(self, index: Index) -> Option[OpaqueParameters]:
        """Returns parameters at index 'index'. They must be bytes-like and will be decoded by the calculator."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples managed by the truth provider."""


class DataManagerInterface(StatelessSurrogateProviderInterface, StatelessTruthProviderInterface, abc.ABC):

    @abc.abstractmethod
    def available_for_calculation(self) -> Sequence[Index]:
        """Returns a sequence of bools: True if no truth calculation is done or is running, else False."""

    @abc.abstractmethod
    def set_result(self, index: Index, value: Option[Target]) -> Result[..., Exception]:
        """Reports the result of a truth simulation."""
