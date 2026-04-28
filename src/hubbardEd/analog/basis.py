"""Basis state generation for the Spin-½ Fermi-Hubbard model."""

from itertools import product
import numpy as np


def basis_states(N: int, L: int):
    """Generate the basis states for the Spin-½ Fermi-Hubbard model with N electrons in L sites.
    Encodes each site with occupation values: 0 (empty), 1 (spin-up), 2 (spin-down), 3 (double occupancy).
    """
    electron_lookup = (0, 1, 1, 2)  # Lookup table for number of electrons in each state
    states = list(product([0, 1, 2, 3], repeat=L))
    valid_states = np.array(
        [state for state in states if sum(electron_lookup[site] for site in state) == N]
    )
    index_map = {tuple(state): idx for idx, state in enumerate(valid_states)}

    return valid_states, index_map
