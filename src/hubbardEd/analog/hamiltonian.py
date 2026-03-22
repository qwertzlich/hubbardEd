"""Solve a 1D chain in the Hubbard model using exact diagonalization"""

import numpy as np
from numpy.typing import NDArray
import scipy.sparse
from .basis import basis_states


def creation_operator(state: NDArray, site: int, spin: int):
    """Create an electron with a given spin at a specific site in the state.
    args:
        state: An array representing the state the operator is applied to
        site: The site where the electron is to be created
        spin: The spin of the electron to be created (1 for up, 2 for down)
    returns:
        A new state with the electron created, or None if the operation is not possible
    """
    if state[site] == spin or state[site] == 3:
        return None
    new_state = state.copy()
    new_state[site] += spin
    return new_state


def annihilation_operator(state: NDArray, site: int, spin: int):
    """Annihilate an electron with a given spin at a specific site in the state.args:
    state: An array representing the state the operator is applied to
    site: The site where the electron is to be annihilated
    spin: The spin of the electron to be annihilated (1 for up, 2 for down)"""
    if state[site] == spin or state[site] == 3:
        new_state = state.copy()
        new_state[site] -= spin
        return new_state
    return None


def check_fermionic_sign(state, site_from, site_to, spin):
    """Calculate the fermionic sign for hopping an electron from site_from to site_to with a given spin."""
    start, end = min(site_from, site_to), max(site_from, site_to)

    count = 0
    # check count for multiple neighbor hopping
    for s in range(start + 1, end):
        if state[s] == 1 or state[s] == 2:
            count += 1
        if state[s] == 3:
            count += 2
    # check count within site in hopping direction
    if spin == 1:  # Up-Spin hopps
        if site_to > site_from and state[site_from] == 3:
            count += 1
        if site_to < site_from and state[site_to] == 2:
            count += 1
    if spin == 2:  # Down-Spin hopps
        if site_to < site_from and state[site_from] == 3:
            count += 1
        if site_to > site_from and state[site_to] == 1:
            count += 1

    return (-1) ** count


def hamiltonian(N: int, L: int, t: float, U: float):
    """Construct the Hamiltonian matrix for the Hubbard model"""
    basis, index_map = basis_states(N, L)
    dim = len(basis)
    HH = scipy.sparse.lil_matrix((dim, dim), dtype=np.float64)

    # On-site interaction
    for basis_state in basis:
        idx = index_map[tuple(basis_state)]
        double_occ = np.sum(basis_state == 3)
        HH[idx, idx] += U * double_occ

    # hopping term (only nearest neighbors)
    for basis_state in basis:
        idx_bra = index_map[tuple(basis_state)]

        for site in range(L):
            target = (site + 1) % L

            for spin_type in [1, 2]:
                if basis_state[site] == spin_type or basis_state[site] == 3:
                    inter = annihilation_operator(basis_state, site, spin_type)
                    if inter is None:
                        continue
                    new_s = creation_operator(inter, target, spin_type)
                    if new_s is not None:
                        idx_ket = index_map[tuple(new_s)]

                        sign = check_fermionic_sign(
                            basis_state, site, target, spin_type
                        )
                        HH[idx_bra, idx_ket] -= t * sign

    if L == 2:  # For L=2, the Hamiltonian is already symmetric
        return HH.tocsr()

    HH = (HH + HH.T - scipy.sparse.diags(HH.diagonal())).tocsr()
    return HH
