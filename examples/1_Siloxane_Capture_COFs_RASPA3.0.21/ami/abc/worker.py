import abc

from ami.abc.calculator import CalculatorInterface
from ami.abc.ranker import RankerInterface

Index = int


class WorkerInterface(CalculatorInterface, RankerInterface, abc.ABC):
    pass
