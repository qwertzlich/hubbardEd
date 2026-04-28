import numpy as np
import scipy.sparse
from ..bitmapped.utils import check_bit, bit_flip

from .types_cav import BasisArray, IndexMap, SparseMatrix


def create_cavity_nn_int_matrix(
    fermion_basis: BasisArray,
    index_map: IndexMap,
    L: int,
    U: float,
    PBC: bool = False,
) -> SparseMatrix:
    """Create the nearest-neighbor interaction matrix for spinless fermions in particle-hole-symmetric form.
    Interaction term: U * sum_i (n_i - 1/2)(n_{i+1} - 1/2)
    Args:
        fermion_basis: The basis states for the spinless fermionic system
        index_map: A dictionary mapping basis states to their indices
        L: The length of the chain
        U: The nearest-neighbor interaction strength
        PBC: Whether to use periodic boundary conditions (default: open boundary)
    Returns:
        The nearest-neighbor interaction terms Hamiltonian matrix in sparse format
    """
    dim = len(fermion_basis)
    HH_nn_int = scipy.sparse.lil_matrix((dim, dim), dtype=np.complex128)

    for basis_state in fermion_basis:
        idx = index_map[basis_state]
        int_term = 0
        end_range = L if PBC else L - 1
        for site in range(end_range):
            next_site = (site + 1) % L
            int_term += (check_bit(basis_state, site) - 0.5) * (
                check_bit(basis_state, next_site) - 0.5
            )
        HH_nn_int[idx, idx] += U * int_term

    return scipy.sparse.csr_matrix(HH_nn_int)


def create_cavity_fermion_hopping_matrix(
    fermion_basis: BasisArray,
    index_map: IndexMap,
    L: int,
    t: float,
    num_fermions: int,
    PBC: bool = False,
) -> SparseMatrix:
    """Create the one-direction hopping matrix for spinless fermions in cavity-coupled model.
    Only right-direction hops are computed; Hermitian conjugate added in full Hamiltonian.
    Includes proper fermionic anticommutation signs.
    args:
        fermion_basis: The basis states for the spinless fermionic system
        index_map: A dictionary mapping basis states to their indices
        L: The length of the chain
        t: The nearest-neighbor hopping amplitude
        num_fermions: The number of fermions in the system (for sign calculation at boundaries)
        PBC: Whether to use periodic boundary conditions
    returns:
        The one-direction fermionic hopping terms Hamiltonian matrix (not yet symmetrized)
    """
    dim = len(fermion_basis)
    HH_ferm_hop = scipy.sparse.lil_matrix((dim, dim), dtype=np.complex128)

    for basis_state in fermion_basis:
        basis_state = basis_state
        idx_bra = index_map[basis_state]

        end_range = L if PBC else L - 1
        for site in range(end_range):
            next_site = (site + 1) % L
            if check_bit(basis_state, site) and not check_bit(basis_state, next_site):
                new_state = bit_flip(bit_flip(basis_state, site), next_site)
                idx_ket = index_map[new_state]
                sign = 1 - 2 * (
                    (site == L - 1) & (num_fermions % 2 == 0)
                )  # Sign change for boundary hop if even number of fermions
                HH_ferm_hop[idx_bra, idx_ket] -= t * sign

    return scipy.sparse.csr_matrix(HH_ferm_hop)


def create_cavity_photonic_displacement_matrix(
    N_photons: int, g: float, L: int
) -> SparseMatrix:
    """Create the photonic displacement operator exp(i * g/sqrt(L) * (a + a_dag)).
    Implements the quantized Peierls phase in Coulomb gauge for cavity coupling.
    args:
        N_photons: The maximum number of photons in the cavity mode
        g: The light-matter coupling strength
        L: The length of the chain (for dipole approximation normalization)
    returns:
        The photonic displacement operator in sparse format
    """
    dim = N_photons + 1
    a = scipy.sparse.diags(np.sqrt(np.arange(1, dim)), offsets=1, dtype=np.complex128)
    a = a.tocsc()
    a_dag = a.T.conj().tocsc()

    HH_shift = scipy.sparse.linalg.expm(1j * g / np.sqrt(L) * (a + a_dag))

    return scipy.sparse.csr_matrix(HH_shift)


def create_cavity_hamiltonian(
    fermion_basis: BasisArray,
    index_map: IndexMap,
    L: int,
    t: float,
    U: float,
    N_photons: int,
    g: float,
    Omega: float,
    pbc: bool = False,
) -> SparseMatrix:
    """Construct the Hamiltonian for spinless fermions with nearest-neighbor interaction coupled to cavity modes.
    Implements quantized Peierls substitution in Coulomb gauge.
    Full Hilbert space: |n_ph> (x) |n_el> = photonic Fock states (x) fermionic electronic basis.
    Args:
        fermion_basis: The basis states for the fermionic system
        index_map: A dictionary mapping fermionic basis states to their indices
        L: The length of the chain
        t: The nearest-neighbor hopping amplitude
        U: The nearest-neighbor interaction strength
        N_photons: The maximum number of photons in the cavity mode
        g: The light-matter coupling strength (normalized by sqrt(L) for dipole approximation)
        Omega: The cavity photon frequency
        pbc: Whether to use periodic boundary conditions
    Returns:
        The full electron-photon Hamiltonian matrix in sparse format
    """
    num_fermions = fermion_basis[0].bit_count()

    HH_nn_int = create_cavity_nn_int_matrix(fermion_basis, index_map, L, U, pbc)

    HH_ferm_hop = create_cavity_fermion_hopping_matrix(
        fermion_basis, index_map, L, t, num_fermions, pbc
    )

    HH_ph_diag = scipy.sparse.diags(
        np.arange(N_photons + 1) * Omega, dtype=np.complex128
    ).tocsr()

    HH_shift = create_cavity_photonic_displacement_matrix(N_photons, g, L)

    I_ph = scipy.sparse.eye(N_photons + 1)
    I_el = scipy.sparse.eye(len(fermion_basis))

    HH = (
        scipy.sparse.kron(I_ph, HH_nn_int)
        + scipy.sparse.kron(HH_ph_diag, I_el)
        + scipy.sparse.kron(HH_shift, HH_ferm_hop)
        + scipy.sparse.kron(HH_shift.T.conj(), HH_ferm_hop.T.conj())
    )

    return scipy.sparse.csr_matrix(HH)
