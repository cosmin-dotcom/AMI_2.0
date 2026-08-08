"""Implements the Result/Either pattern.

* Result[T, E] is the base type generic over T. It must be used in type declarations.
* Ok[T](x: T) represents a correct value x of type T.
* Err[E](e: E) represents an error e of type E.

E.g.

    def divide(a: float, b: float) -> Result[float, str]:
        if b == 0.0:
            return Err("Division by zero is forbidden.")
        return Ok(a / b)

    val = divide(1.0, 2.0)

    match val:
        case Ok(x):
            print(f"{a}/{b} = {x}")
        case Err(msg):
            raise TypeError(msg)

    # Or

    print(f"{a}/{b} = {val.unwrap()}")
"""

from dataclasses import dataclass
from typing import Generic, TypeVar, Callable, final

T = TypeVar('T')
E = TypeVar('E')


class Result(Generic[T, E]):
    def __new__(cls, value) -> "Result":
        if cls is Result:
            raise RuntimeError("'Result' cannot be directly instanciated. Please use 'Ok(...)' or 'Err(...)'.")
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
class Ok(Result[T, E]):
    value: T

    def __repr__(self) -> str:
        return f'Ok({self.value})'

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
@dataclass(slots=True, frozen=True, repr=False)
class Err(Result[T, E]):
    error: E

    def __repr__(self) -> str:
        return f'Err({self.error})'

    def expect(self, err_msg: str) -> T:
        raise TypeError(err_msg)

    def unwrap(self) -> T:
        if isinstance(self.error, Exception):
            raise self.error
        raise TypeError("Value is incorrect.\nError: {repr(self.error)}")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, default_fn: Callable[[], T]) -> T:
        return default_fn()

    def __bool__(self):
        return False
