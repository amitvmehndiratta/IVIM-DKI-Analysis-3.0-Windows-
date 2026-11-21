"""MLIR NVGPU dialect."""

from typing import Final

from jaxlib.mlir import ir

import jaxlib.jaxlib.mlir.ir


class TensorMapDescriptorType(jaxlib.ir.Type):
    def __init__(self, cast_from_type: jaxlib.ir.Type) -> None: ...

    static_typeid: Final[jaxlib.ir.TypeID] = ...
    """(arg: object, /) -> jaxlib.ir.TypeID"""

    @property
    def typeid(self) -> jaxlib.ir.TypeID: ...

    def __repr__(self) -> str: ...

    type_name: Final[str] = ...
    """(arg: object, /) -> str"""

    @staticmethod
    def get(tensor_type: jaxlib.ir.Type, swizzle: int, l2promo: int, oob_fill: int, interleave: int, context: ir.Context | None = None) -> TensorMapDescriptorType:
        """Gets an instance of TensorMapDescriptorType in the same context"""
