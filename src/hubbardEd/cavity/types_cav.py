"""Shared type aliases for the cavity submodule."""

from __future__ import annotations

import scipy.sparse
from numpy.typing import NDArray
from typing import TypeAlias

import numpy as np


FermionBasisState: TypeAlias = int
BasisArray: TypeAlias = NDArray[np.int64]
IndexMap: TypeAlias = dict[FermionBasisState, int]
StateVector: TypeAlias = NDArray[np.complex128]
SparseMatrix: TypeAlias = scipy.sparse.csr_matrix