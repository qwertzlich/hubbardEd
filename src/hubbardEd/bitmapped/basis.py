"""This module calculates the basis states for the hubbard model in bitmapped representation."""

import numpy as np

from .types import BasisArray, IndexMap


def bitmap_basis_states(
    num_sites: int, num_up: int, num_down: int
) -> tuple[BasisArray, IndexMap]:
    """Generate the basis states for the Spin-½ Fermi-Hubbard model in bitmapped representation.
    Each basis state is a tuple (up_bits, down_bits) where each bit encodes occupation at a site.
    args:
        num_sites: The total number of sites in the system
        num_up: The number of spin-up electrons
        num_down: The number of spin-down electrons
    returns:
        A tuple of (basis_states array, index_map dict) for the fixed (N_up, N_down) sector.
    """
    up_states = [i for i in range(1 << num_sites) if bin(i).count("1") == num_up]
    down_states = [i for i in range(1 << num_sites) if bin(i).count("1") == num_down]
    basis = [(i, j) for i in up_states for j in down_states]

    index_map = {state: idx for idx, state in enumerate(basis)}

    return np.array(basis, dtype=np.int64), index_map
