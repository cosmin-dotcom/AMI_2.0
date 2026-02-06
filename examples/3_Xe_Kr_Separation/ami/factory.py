from dataclasses import dataclass, field, Field, fields, is_dataclass, MISSING
from typing import Iterator, MutableMapping, Any, ClassVar

import ami.abc
from ami.abc.factory import BuiltObject
from ami.result import Result, Ok, Err

def optional_field(field):
    if not field.init:
        return True
    if field.default is not MISSING:
        return True
    if field.default_factory is not MISSING:
        return True
    return False

@dataclass(slots=True, frozen=True)
class DataclassFactory(ami.abc.FactoryInterface[BuiltObject]):
    dataclass: ClassVar = NotImplemented  # ClassVar[BuiltObject]

    _MISSING = object()
    _OPTIONAL = object()
    _objects: MutableMapping[str, Any] = field(default_factory=dict)


    def __post_init__(self):
        for f in self._fields():
            if f not in self._objects:

                if not f.init:
                    continue
                if optional_field(f):
                    self._objects[f.name] = self._OPTIONAL
                    continue
                self._objects[f.name] = self._MISSING
        if not is_dataclass(self.dataclass):
            raise TypeError("Class var 'dataclass' must be a dataclass.")

    def _is_finalized(self) -> bool:
        for f in self._fields():
            if optional_field(f) or not f.init:
                continue
            if self._objects.get(f.name) is self._MISSING:
                return False
        return True

    def _missing(self) -> Iterator[str]:
        r = []
        for f in self._fields():
            if self._objects[f.name] is self._MISSING:
                r.append(f.name)
        return r

    def set(self, key: str, value: Any):
        if key not in self._objects:
            raise KeyError(f"Invalid key '{key}' for factory '{self.__class__.__name__}'.")
        self._objects[key] = value

    @classmethod
    def _fields(cls) -> Iterator[Field]:
        return tuple(fields(cls.dataclass))

    def build(self) -> Result[BuiltObject, TypeError]:
        if not self._is_finalized():
            missing = self._missing()
            msg = f"The following fields are missing: {', '.join(missing)}."
            return Err(TypeError(msg))
        kwargs = {k: v for k, v in self._objects.items() if v is not self._OPTIONAL}
        return Ok(self.dataclass(**kwargs))
