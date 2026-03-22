"""This module calculates the basis states for the hubbard model in bitmapped representation."""

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias


BasisState: TypeAlias = tuple[int, int]
BasisArray: TypeAlias = NDArray[np.int64]
IndexMap: TypeAlias = dict[BasisState, int]


def bitmap_basis_states(
    num_sites: int, num_up: int, num_down: int
) -> tuple[BasisArray, IndexMap]:
    """Generate the basis states for a given number of sites and electrons in bitmapped representation.
    args:
        num_sites: The total number of sites in the system
        num_up: The number of spin-up electrons
        num_down: The number of spin-down electrons
    returns:
        An array of basis states represented as integers, where each bit corresponds to a site and its occupation.
    """
    up_states = [i for i in range(1 << num_sites) if bin(i).count("1") == num_up]
    down_states = [i for i in range(1 << num_sites) if bin(i).count("1") == num_down]
    basis = [(i, j) for i in up_states for j in down_states]

    index_map = {state: idx for idx, state in enumerate(basis)}

    return np.array(basis, dtype=np.int64), index_map
