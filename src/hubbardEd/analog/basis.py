"""Module for generating the basis states for the Hubbard model."""

from itertools import product
import numpy as np


def basis_states(N: int, L: int):
    """Generate the basis states for N electrons in L sites
    0 -> empty, 1 -> Spin Up, 2 -> Spin Down, 3 -> Double Occupancy
    """
    electron_lookup = (0, 1, 1, 2)  # Lookup table for number of electrons in each state
    states = list(product([0, 1, 2, 3], repeat=L))
    valid_states = np.array(
        [state for state in states if sum(electron_lookup[site] for site in state) == N]
    )
    index_map = {tuple(state): idx for idx, state in enumerate(valid_states)}

    return valid_states, index_map
