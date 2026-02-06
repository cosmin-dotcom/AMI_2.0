"""Implements the Maybe/Option pattern.

* Option[T] is the base type generic over T. It must be used in type declarations.
* Some[T](x: T) represents an existing value x of type T.
* Nothing represents the lack of value.

E.g.

    def divide(a: float, b: float) -> Option[float]:
        if b == 0.0:
            return Nothing
        return Some(a / b)

    val = divide(1.0, 2.0)

    match val:
        case Some(x):
            print(f"{a}/{b} = {x}")
        case Nothing:
            raise TypeError("Division by zero is forbidden.")

    # Or

    print(f"{a}/{b} = {val.expect("Division by zero is forbidden.")}")
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, final

T = TypeVar('T')


class Option(Generic[T]):
    def __new__(cls, value) -> "Option":
        if cls is Option:
            raise RuntimeError("'Option' cannot be directly instanciated. Please use 'Some(...)' or 'Nothing'.")
        obj = super().__new__(cls)
        return obj

    def expect(self, err_msg: str) -> T:
        raise NotImplementedError()

    def unwrap(self) -> T:
        raise NotImplementedError()

    def unwrap_or(self, default: T) -> T:
        raise NotImplementedError()

    def unwrap_or_else(self, default_fn: Callable[[], T]) -> T:
        raise NotImplementedError()


@final
@dataclass(slots=True, frozen=True, repr=False)
class Some(Option[T]):
    value: T

    def __repr__(self) -> str:
        return f'Some({self.value})'

    def expect(self, err_msg: str) -> T:
        return self.value

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, default_fn: Callable[[], T]) -> T:
        return self.value

    def __bool__(self):
        return True


@final
class Nothing(Option[T]):

    def __repr__(self) -> str:
        return 'Nothing'

    def expect(self, err_msg: str) -> T:
        raise TypeError(err_msg)

    def unwrap(self) -> T:
        raise TypeError("No value to unwrap.")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, default_fn: Callable[[], T]) -> T:
        return default_fn()

    def __bool__(self):
        return False


# This is normal and expected: we need a singleton instance.
Nothing = Nothing(None)
