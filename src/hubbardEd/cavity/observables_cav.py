"""Observable operators for spinless fermions coupled to cavity modes."""

import numpy as np

from .types_cav import IndexMap, StateVector


def site_correlation(
    psi: StateVector,
    site_i: int,
    site_j: int,
    index_map: IndexMap,
    N_photons: int,
) -> float:
    """Compute the two-point spinless fermion correlation <n_i n_j> in the coupled electron-photon Hilbert space.
    Args:
        psi: The state vector in the full electron-photon Hilbert space
        site_i: The first fermionic site index
        site_j: The second fermionic site index
        index_map: A dictionary mapping fermionic basis states to their indices
        N_photons: The maximum number of photons in the cavity mode
    Returns:
        The long-range charge correlation <n_i n_j> expectation value, summed over photon states
    """
    correlation = 0.0
    for photon_num in range(N_photons + 1):
        for basis_state, idx in index_map.items():
            n_i = (basis_state >> site_i) & 1  # Occupation at site i
            n_j = (basis_state >> site_j) & 1  # Occupation at site j
            correlation += (
                n_i * n_j * np.abs(psi[idx + photon_num * len(index_map)]) ** 2
            )
    return correlation
