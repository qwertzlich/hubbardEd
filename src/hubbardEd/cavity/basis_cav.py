"""Basis state generation for spinless fermions coupled to cavity modes."""

import numpy as np

from .types_cav import BasisArray, IndexMap


def spinless_fermion_basis(
    num_sites: int, num_fermions: int
) -> tuple[BasisArray, IndexMap]:
    """Generate the fixed-particle-number basis for spinless fermions in bitmapped representation.
    Only the fermionic basis is generated; photonic Fock space is constructed during Hamiltonian assembly.
    args:
        num_sites: The total number of sites in the system
        num_fermions: The number of spinless fermions to fix in the system
    returns:
        A tuple of (basis_states array, index_map dict) where basis states are single integers representing fermion occupation.
    """
    states = [i for i in range(1 << num_sites) if bin(i).count("1") == num_fermions]
    index_map = {state: idx for idx, state in enumerate(states)}
    return np.array(states, dtype=np.int64), index_map
