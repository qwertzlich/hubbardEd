"""Shared type aliases for the bitmapped submodule."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import scipy.sparse
from numpy.typing import NDArray


BasisState: TypeAlias = tuple[int, int]
BasisArray: TypeAlias = NDArray[np.int64]
IndexMap: TypeAlias = dict[BasisState, int]
StateVector: TypeAlias = NDArray[np.complex128]
SparseMatrix: TypeAlias = scipy.sparse.csr_matrix
Spin: TypeAlias = Literal[1, 2]
GaugeChoice: TypeAlias = Literal["velocity", "length"]
GaugeField: TypeAlias = Callable[[float], float]


class HubbardParams(TypedDict):
    L: int
    PBC: bool
    e_charge: float
    hbar: float


class HubbardSystem(TypedDict):
    H_int: SparseMatrix
    H_hop: SparseMatrix
    R_op: SparseMatrix
    basis: BasisArray
    index_map: IndexMap
    Params: HubbardParams
