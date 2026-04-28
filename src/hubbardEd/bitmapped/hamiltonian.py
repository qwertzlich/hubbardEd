"""This module defines the Hamiltonian for the Hubbard model using a bitmapped representation of the basis states."""

import numpy as np
from numpy.typing import NDArray
import scipy.sparse
from .utils import bit_flip, check_bit

from .types import BasisArray, BasisState, IndexMap, SparseMatrix, Spin


def bm_hopping_operator(
    state: BasisState, site_from: int, site_to: int, spin: Spin
) -> BasisState | None:
    """Apply fermionic hopping operator for the Spin-½ Fermi-Hubbard model in bitmapped representation.
    Args:
        state: A tuple of (up_bits, down_bits) representing occupation of up and down spins
        site_from: The site from which the electron hops
        site_to: The site to which the electron hops
        spin: The spin state (1 for up, 2 for down)
    Returns:
        A new state after hopping, or None if the operation violates occupation constraints
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
    state: BasisState,
    site_from: int,
    site_to: int,
    spin: Spin,
    is_boundary_hop: bool = False,
) -> int:
    """Calculate the fermionic anticommutation sign for hopping in the Spin-½ Fermi-Hubbard model.
    Counts occupied sites of the same spin between source and target to determine sign.
    Args:
        state: A tuple of (up_bits, down_bits) representing occupation of up and down spins
        site_from: The site from which the electron hops
        site_to: The site to which the electron hops
        spin: The spin state (1 for up, 2 for down)
        is_boundary_hop: Whether the hop is across the boundary (for periodic boundary conditions)
    Returns:
        The fermionic sign (+1 or -1) for the hopping process
    """
    up_state, down_state = state
    start, end = min(site_from, site_to), max(site_from, site_to)

    if is_boundary_hop:
        num_passed = (state[spin - 1]).bit_count()
        return (-1) ** (num_passed - 1)

    mask = ((1 << end) - 1) ^ ((1 << (start + 1)) - 1)

    if spin == 1:
        num_passed = (up_state & mask).bit_count()
    else:
        num_passed = (down_state & mask).bit_count()
    return (-1) ** num_passed


def create_interaction_matrix(
    basis: BasisArray, index_map: IndexMap, U: float
) -> SparseMatrix:
    """Create the on-site interaction matrix for the Spin-½ Fermi-Hubbard model.
    Diagonal matrix with U * (number of doubly-occupied sites) on each diagonal element.
    Args:
        basis: The basis states for the system
        index_map: A dictionary mapping basis states to their indices
        U: The on-site interaction strength
    Returns:
        The on-site interaction terms Hamiltonian matrix
    """
    dim = len(basis)
    HH = scipy.sparse.lil_matrix((dim, dim), dtype=np.complex128)
    for basis_state in basis:
        idx = index_map[tuple(basis_state)]
        up_state, down_state = basis_state
        double_occ = (up_state & down_state).bit_count()
        HH[idx, idx] += U * double_occ
    return scipy.sparse.csr_matrix(HH)


def create_hopping_matrix(
    basis: BasisArray,
    index_map: IndexMap,
    L: int,
    t: float,
    PBC: bool = True,
) -> SparseMatrix:
    """Create the nearest-neighbor hopping matrix for the Spin-½ Fermi-Hubbard model.
    Includes proper fermionic anticommutation signs.
    Args:
        basis: The basis states for the system
        index_map: A dictionary mapping basis states to their indices
        L: The length of the chain
        t: The nearest-neighbor hopping amplitude
        PBC: Whether to use periodic boundary conditions
    Returns:
        The hopping terms Hamiltonian matrix (not yet symmetrized for Hermiticity)
    """
    dim = len(basis)
    HH_hop = scipy.sparse.lil_matrix((dim, dim), dtype=np.complex128)

    for basis_state in basis:
        basis_state = tuple(basis_state)
        idx_bra = index_map[basis_state]

        for target in range(L):
            site = (target + 1) % L
            is_boundary_hop = (target == L - 1) and (site == 0)
            if not PBC and is_boundary_hop:
                continue

            for spin in (1, 2):  # Spin up and down
                new_state = bm_hopping_operator(basis_state, site, target, spin)
                if new_state is None:
                    continue
                idx_ket = index_map[new_state]
                sign = check_fermionic_sign(
                    basis_state, site, target, spin, is_boundary_hop=is_boundary_hop
                )

                HH_hop[idx_bra, idx_ket] -= t * sign

    return scipy.sparse.csr_matrix(HH_hop)


def bm_create_base_hamiltonian(
    basis: BasisArray,
    index_map: IndexMap,
    L: int,
    t: float,
    U: float,
    PBC: bool = True,
) -> SparseMatrix:
    """Construct the full Hamiltonian matrix for the Spin-½ Fermi-Hubbard model.
    Combines on-site interaction and nearest-neighbor hopping terms with proper symmetrization.
    Args:
        basis: The basis states for the system
        index_map: A dictionary mapping basis states to their indices
        L: The length of the chain
        t: The nearest-neighbor hopping amplitude
        U: The on-site interaction strength
        PBC: Whether to use periodic boundary conditions
    Returns:
        The full Hamiltonian matrix in sparse format
    """

    HH_int = create_interaction_matrix(basis, index_map, U)

    # Add hopping terms (only to the right) and then symmetrize the Hamiltonian
    # Also include the velocity gauge if field is non-zero
    HH_hop = create_hopping_matrix(basis, index_map, L, t, PBC)

    HH = HH_int + HH_hop

    if L == 2 and PBC:
        return scipy.sparse.csr_matrix(
            HH
        )  # Special case for L=2 with PBC, no need to symmetrize

    HH = HH + HH_hop.T.conj()  # Symmetrize the Hamiltonian to include left hops

    return scipy.sparse.csr_matrix(HH)


def bm_get_eigenstates(
    HH: scipy.sparse.spmatrix, num_evals: int
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Compute low-lying eigenvalues and eigenvectors using sparse Lanczos diagonalization.
    Returns sorted in ascending order of energy.
    """
    evals, evecs = scipy.sparse.linalg.eigsh(HH, k=num_evals, which="SA")
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]
