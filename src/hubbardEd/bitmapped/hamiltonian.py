"""This module defines the Hamiltonian for the Hubbard model using a bitmapped representation of the basis states."""

import numpy as np
import scipy.sparse
from hubbardEd.bitmapped import bitmap_basis_states, bit_flip, check_bit


def bm_hopping_operator(
    state: tuple, site_from: int, site_to: int, spin: int
) -> tuple | None:
    """Hopping operator for the Hubbard model in bitmapped representation
    args:
        state: A tuple of two integers representing the occupation of up and down spins
        site_from: The site from which the electron hops
        site_to: The site to which the electron hops
        spin: The spin state (1 for up, 2 for down)
    returns:
        A new state after hopping, or None if the operation is not possible
    """
    up_state, down_state = state
    if spin == 1:
        if check_bit(up_state, site_from) and not check_bit(up_state, site_to):
            new_up_state = bit_flip(bit_flip(up_state, site_from), site_to)
            return (new_up_state, down_state)
    elif spin == 2:
        if check_bit(down_state, site_from) and not check_bit(down_state, site_to):
            new_down_state = bit_flip(bit_flip(down_state, site_from), site_to)
            return (up_state, new_down_state)
    return None


def check_fermionic_sign(
    state: tuple, site_from: int, site_to: int, spin: int, PBC: bool = False
) -> int:
    """Calculate the fermionic sign for hopping an electron from site_from to site_to in a given state
    args:
        state: A tuple of two integers representing the occupation of up and down spins
        site_from: The site from which the electron hops
        site_to: The site to which the electron hops
        spin: The spin state (1 for up, 2 for down)
        PBC: Wether ot not the hop is across the boundary (for periodic boundary conditions)
    returns:
        The fermionic sign (+1 or -1) for the hopping process
    """
    up_state, down_state = state
    start, end = min(site_from, site_to), max(site_from, site_to)

    if PBC:
        num_passed = (state[spin - 1]).bit_count()
        return (-1) ** (num_passed - 1)

    mask = ((1 << end) - 1) ^ ((1 << (start + 1)) - 1)

    if spin == 1:
        num_passed = (up_state & mask).bit_count()
    else:
        num_passed = (down_state & mask).bit_count()
    return (-1) ** num_passed


def _add_interaction_terms(HH, basis, index_map, U):
    """Calculate the Matrix element for the HH interaction term for a given state
    args:
            state: A tuple of two integers representing the occupation of up and down spins
    returns:
        The matrix element for the HH interaction term"""
    for basis_state in basis:
        idx = index_map[tuple(basis_state)]
        up_state, down_state = basis_state
        double_occ = (up_state & down_state).bit_count()
        HH[idx, idx] += U * double_occ


def _add_hopping_terms(HH, basis, index_map, L, t: float):
    """Helper function to add nearest neighbor hopping terms to the Hamiltonian matrix."""
    for basis_state in basis:
        basis_state = tuple(basis_state)
        idx_bra = index_map[basis_state]

        for target in range(L):
            site = (target + 1) % L

            for spin in [1, 2]:  # Spin up and down
                new_state = bm_hopping_operator(basis_state, site, target, spin)
                if new_state is None:
                    continue
                idx_ket = index_map[new_state]
                is_boundary_hop = (site == L - 1) and (target == 0)
                sign = check_fermionic_sign(
                    basis_state, site, target, spin, PBC=is_boundary_hop
                )

                HH[idx_bra, idx_ket] -= t * sign


def bm_create_hamiltonian(N_up: int, N_down: int, L: int, t: float, U: float):
    """Construct the Hamiltonian matrix for the Hubbard model. Only hopps to the right are calculated and HH later symmetrized."""
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    dim = len(basis)
    HH = scipy.sparse.lil_matrix((dim, dim), dtype=np.float64)

    _add_interaction_terms(HH, basis, index_map, U)
    _add_hopping_terms(HH, basis, index_map, L, t)

    if L > 2:  # L=2 is a special case
        HH = HH + HH.T - scipy.sparse.diags(HH.diagonal())
    return HH.tocsr()


def bm_get_eigenstates(HH):
    """Return the eigenvalues and vectors of the given HH matrix, sorted in ascending order."""
    evals, evecs = scipy.sparse.linalg.eigsh(HH, k=HH.shape[0] - 1, which="SA")
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]
