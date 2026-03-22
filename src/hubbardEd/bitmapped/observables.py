"""This module contains operators representing observables which may be used to calculate expectation values"""

import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias
from .utils import check_bit


BasisState: TypeAlias = tuple[int, int]
BasisArray: TypeAlias = NDArray[np.int64]
StateVector: TypeAlias = NDArray[np.complex128]


def get_doublon_expectation(
    psi: StateVector, basis_states: BasisArray, L: int
) -> list[float]:
    """Calculate the expectation value of doublon number operator per site for a given state psi and basis states.
    Args:
        psi: State vector in the given basis
        basis_states: List of basis states corresponding to the rows of psi
        L: The number of sites in the system
    Returns:
        The expectation value of the doublon number operator
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
    """Calculate the expectation value of the occupation number correlation between the two most distant sites for a given state psi and basis states.
    Args:
        psi: State vector in the given basis
        basis_states: List of basis states corresponding to the rows of psi
        L: The number of sites in the system
    Returns:
        The expectation value of the occupation number correlation between the two most distant sites
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    expectation = 0.0  # <n_1 n_L/2 >
    for idx, state in enumerate(basis_states):
        n1 = check_bit(state[0], 0) + check_bit(state[1], 0)
        nL = check_bit(state[0], L // 2) + check_bit(state[1], L // 2)
        expectation += n1 * nL * abs(psi[idx]) ** 2
    return expectation
