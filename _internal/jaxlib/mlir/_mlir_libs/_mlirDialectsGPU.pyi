"""MLIR GPU Dialect"""

from typing import Final

from jaxlib.mlir import ir

import jaxlib.jaxlib.mlir.ir


class AsyncTokenType(jaxlib.ir.Type):
    def __init__(self, cast_from_type: jaxlib.ir.Type) -> None: ...

    static_typeid: Final[jaxlib.ir.TypeID] = ...
    """(arg: object, /) -> jaxlib.ir.TypeID"""

    @property
    def typeid(self) -> jaxlib.ir.TypeID: ...

    def __repr__(self) -> str: ...

    type_name: Final[str] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(context: ir.Context | None = None) -> AsyncTokenType:
        """Gets an instance of AsyncTokenType in the same context"""

class ObjectAttr(jaxlib.ir.Attribute):
    def __init__(self, cast_from_attr: jaxlib.ir.Attribute) -> None: ...

    @property
    def type(self) -> jaxlib.ir.Type: ...

    static_typeid: Final[jaxlib.ir.TypeID] = ...
    """(arg: object, /) -> jaxlib.ir.TypeID"""

    @property
    def typeid(self) -> jaxlib.ir.TypeID: ...

    def __repr__(self) -> str: ...

    attr_name: Final[str] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(target: jaxlib.ir.Attribute, format: int, object: bytes, properties: jaxlib.ir.DictAttr | None = None, kernels: jaxlib.ir.Attribute | None = None, context: ir.Context | None = None) -> ObjectAttr:
        """Gets a gpu.object from parameters."""

    @property
    def target(self) -> jaxlib.ir.Attribute: ...

    @property
    def format(self) -> int: ...

    @property
    def object(self) -> bytes: ...

    @property
    def properties(self) -> jaxlib.ir.DictAttr | None: ...

    @property
    def kernels(self) -> jaxlib.ir.Attribute | None: ...
