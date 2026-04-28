"""Observable operators for the Spin-½ Fermi-Hubbard model with bitmapped basis representation."""

import numpy as np

from .types import BasisArray, StateVector
from .utils import check_bit


def get_doublon_expectation(
    psi: StateVector, basis_states: BasisArray, L: int
) -> list[float]:
    """Calculate site-resolved doublon expectation values for the Spin-½ Fermi-Hubbard model.
    Returns the probability of finding both spins occupied at each site.
    Args:
        psi: State vector in the given basis
        basis_states: List of basis states (up_bits, down_bits) tuples
        L: The number of sites in the system
    Returns:
        List of doublon expectation values <n_{i↑} n_{i↓}> for each site i
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)  # ensure psi is a 1D array
    expectation = np.zeros(L, dtype=float)

    for idx, (up_state, down_state) in enumerate(basis_states):  # each dimension in psi
        doublon_mask = int(up_state) & int(down_state)
        if doublon_mask == 0:
            continue

        weight = float(np.abs(psi[idx]) ** 2)

        # iterate over all set bits in doublon_mask
        while doublon_mask:
            lsb = doublon_mask & -doublon_mask  # keeps the rightmost set bit
            site = lsb.bit_length() - 1
            expectation[site] += weight
            doublon_mask ^= lsb  # remove the rightmost set bit from mask

    return expectation.tolist()


def get_LRC_expectation(psi: StateVector, basis_states: BasisArray, L: int) -> float:
    """Calculate the long-range charge correlation <n_0 n_{L/2}> for the Spin-½ Fermi-Hubbard model.
    Computes the correlation between site 0 and the middle site (L/2).
    Args:
        psi: State vector in the given basis
        basis_states: List of basis states (up_bits, down_bits) tuples
        L: The number of sites in the system
    Returns:
        The expectation value of <n_0 n_{L/2}> (total occupation at both sites)
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    expectation = 0.0  # <n_1 n_L/2 >
    for idx, state in enumerate(basis_states):
        n1 = check_bit(state[0], 0) + check_bit(state[1], 0)
        nL = check_bit(state[0], L // 2) + check_bit(state[1], L // 2)
        expectation += n1 * nL * abs(psi[idx]) ** 2
    return expectation
