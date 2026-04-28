"""Time propagation for the Spin-½ Fermi-Hubbard model under external gauge fields using bitmapped representation."""

import numpy as np
from scipy.constants import hbar, e
import scipy.sparse
from .basis import bitmap_basis_states
from .hamiltonian import create_hopping_matrix, create_interaction_matrix
from .utils import check_bit

from .types import (
    BasisArray,
    GaugeChoice,
    GaugeField,
    HubbardSystem,
    IndexMap,
    SparseMatrix,
    StateVector,
)


def create_position_matrix(
    basis: BasisArray, index_map: IndexMap, L: int
) -> SparseMatrix:
    """Create the position operator diagonal matrix sum_i (i * n_i) for both spins.
    Used in length gauge formulation of external fields.
    """
    dim = len(basis)
    R_op = scipy.sparse.lil_matrix((dim, dim), dtype=np.complex128)

    for basis_state in basis:
        idx = index_map[tuple(basis_state)]
        up_state, down_state = basis_state
        val = 0
        for site in range(L):
            if check_bit(up_state, site):
                val += site
            if check_bit(down_state, site):
                val += site
        R_op[idx, idx] = val

    return scipy.sparse.csr_matrix(R_op)


def setup_hubbard_system(
    N_up: int,
    N_down: int,
    L: int,
    t: float,
    U: float,
    PBC: bool = True,
    e_charge: float = e,
    hbar_val: float = hbar,
) -> HubbardSystem:
    """Set up the Spin-½ Fermi-Hubbard Hamiltonian and related operators for time evolution.
    Precomputes interaction, hopping, and position operators.
    args:
        N_up: Number of spin-up electrons
        N_down: Number of spin-down electrons
        L: Number of lattice sites
        t: Nearest-neighbor hopping amplitude
        U: On-site interaction strength
        PBC: Whether to use periodic boundary conditions
        e_charge: Elementary charge (for gauge field coupling)
        hbar_val: Planck constant (for dynamics)
    returns:
        Dictionary containing precomputed operators (H_int, H_hop, R_op, basis, index_map, parameters)
    """
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    HH_int = create_interaction_matrix(basis, index_map, U)
    HH_hop = create_hopping_matrix(basis, index_map, L, t, PBC)
    R_op = create_position_matrix(basis, index_map, L)
    return {
        "H_int": HH_int,
        "H_hop": HH_hop,
        "R_op": R_op,
        "basis": basis,
        "index_map": index_map,
        "Params": {"L": L, "PBC": PBC, "e_charge": e_charge, "hbar": hbar_val},
    }


def time_evolve_state(
    psi: StateVector,
    system: HubbardSystem,
    gauge_choice: GaugeChoice,
    gauge_field: GaugeField,
    t0: float,
    tf: float,
    num_points: int,
) -> StateVector:
    """Time-evolve a quantum state under the Spin-½ Fermi-Hubbard Hamiltonian with external gauge fields.
    Supports velocity and length gauges for electric field coupling.
    Returns the evolved state at time tf.
    """
    params = system["Params"]

    time_points = np.linspace(t0, tf, num_points, endpoint=False)
    dt = time_points[1] - time_points[0]

    psi = np.asarray(psi, dtype=np.complex128).copy()
    for t in time_points:
        t_mid = t + dt / 2
        gauge_val = gauge_field(t_mid)

        if gauge_choice == "velocity":
            HH_velocity = system["H_hop"].copy()
            phi = params["e_charge"] * gauge_val / params["hbar"]
            HH_velocity.data *= np.exp(
                1j * phi
            )  # Because of bitmapping the hops are in negative x direction, therefore the sign of charge and dl cancel out

            HH_t = system["H_int"] + HH_velocity + HH_velocity.T.conj()

        elif gauge_choice == "length":
            if params["PBC"]:
                raise ValueError(
                    "Length gauge is not compatible with periodic boundary conditions."
                )
            potential = params["e_charge"] * gauge_val * system["R_op"]
            HH_t = (
                system["H_int"] + system["H_hop"] + system["H_hop"].T.conj() - potential
            )

        else:
            raise ValueError(
                f"Invalid gauge choice. Must be 'velocity' or 'length'. Not {gauge_choice}"
            )
        A = (-1j / params["hbar"]) * HH_t.tocsc() * dt
        psi = scipy.sparse.linalg.expm_multiply(A, psi)

    return psi
