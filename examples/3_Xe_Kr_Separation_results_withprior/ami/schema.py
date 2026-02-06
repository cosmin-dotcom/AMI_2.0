from dataclasses import dataclass
from typing import Type, Tuple, Iterator

from ami.abc import SchemaInterface


@dataclass(frozen=True, slots=True)
class Schema(SchemaInterface):
    input_schema: Iterator[Tuple[str, Type]]
    output_schema: Iterator[Tuple[str, Type]]

    def feature(self) -> Iterator[Tuple[str, Type]]:
        return self.input_schema

    def target(self) -> Iterator[Tuple[str, Type]]:
        return self.output_schema
