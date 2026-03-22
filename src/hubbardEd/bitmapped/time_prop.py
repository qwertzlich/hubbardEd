"""This module contains functions for time propagation of given quantum states under the Hubbard Hamiltonian using a bitmapped representation"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Literal, TypeAlias, TypedDict
from scipy.constants import hbar, e
import scipy.sparse
from .basis import bitmap_basis_states
from .hamiltonian import create_hopping_matrix, create_interaction_matrix
from .utils import check_bit


BasisState: TypeAlias = tuple[int, int]
BasisArray: TypeAlias = NDArray[np.int64]
IndexMap: TypeAlias = dict[BasisState, int]
SparseMatrix: TypeAlias = scipy.sparse.csr_matrix
GaugeChoice: TypeAlias = Literal["velocity", "length"]
GaugeField: TypeAlias = Callable[[float], float]
StateVector: TypeAlias = NDArray[np.complex128]


class HubbardParams(TypedDict):
    L: int
    PBC: bool
    e_charge: float
    hbar: float


class HubbardSystem(TypedDict):
    H_int: SparseMatrix
    H_hop: SparseMatrix
    R_op: SparseMatrix
    basis: BasisArray
    index_map: IndexMap
    Params: HubbardParams


def create_position_matrix(
    basis: BasisArray, index_map: IndexMap, L: int
) -> SparseMatrix:
    """Creates a diagonal matrix representing the position operator sum(i * n_i)."""
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
    """Set up the Hubbard Hamiltonian matrix for a given system configuration.
    args:
        N_up: Number of spin-up electrons
        N_down: Number of spin-down electrons
        L: Number of lattice sites
        t: Hopping amplitude
        U: On-site interaction strength
        PBC: Whether to use periodic boundary conditions
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
    """Time evolve a given state under the time dependent Hubbard Hamiltonian."""
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
