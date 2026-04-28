"""Testing routines for the bitmapped Hamiltonian."""

import os
import sys
import numpy as np
import pytest

directory = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(directory, "..", "src")))

from hubbardEd.bitmapped import (
    bitmap_basis_states,
    bm_hopping_operator,
    check_fermionic_sign,
    bm_create_base_hamiltonian,
    bit_flip,
    check_bit,
    setup_hubbard_system,
    time_evolve_state,
    get_doublon_expectation,
    bm_get_eigenstates,
)


@pytest.mark.parametrize(
    "bin_state, site, expected",
    [
        (0b1010, 1, 0b1000),
        (0b1010, 2, 0b1110),
        (0b1010, 3, 0b0010),
    ],
)
def test_bit_flip(bin_state, site, expected):
    """Test the bit_flip function."""
    assert bit_flip(bin_state, site) == expected


@pytest.mark.parametrize(
    "bin_state, site, expected",
    [
        (0b1010, 1, True),
        (0b1010, 2, False),
        (0b1010, 3, True),
    ],
)
def test_check_bit(bin_state, site, expected):
    """Test the check_bit function."""
    assert check_bit(bin_state, site) == expected


@pytest.mark.parametrize(
    "L, up, down, expected_len", [(2, 2, 1, 2), (2, 1, 1, 4), (3, 2, 1, 9)]
)
def test_bitmapped_basis(L, up, down, expected_len):
    """Test the generation of basis states in bitmapped representation."""
    basis, _ = bitmap_basis_states(L, up, down)
    assert (
        len(basis) == expected_len
    ), f"Expected {expected_len} basis states, got {len(basis)}"


@pytest.mark.parametrize(
    "state, site_from, site_to, spin, expected",
    [
        ((0b10, 0b00), 1, 0, 1, (0b01, 0b00)),
        ((0b11, 0b01), 0, 1, 2, (0b11, 0b10)),
        ((0b01, 0b11), 0, 1, 2, None),
        ((0b00, 0b11), 0, 1, 1, None),
    ],
)
def test_bm_hopping_operator(state, site_from, site_to, spin, expected):
    """Test the hopping operator."""
    if expected is None:
        assert bm_hopping_operator(state, site_from, site_to, spin) is None
    else:
        np.testing.assert_array_equal(
            bm_hopping_operator(state, site_from, site_to, spin), expected
        )


@pytest.mark.parametrize(
    "state, site_from, site_to, spin, is_boundary_hop, expected",
    [
        ((0b10, 0b00), 1, 0, 1, False, 1),
        ((0b110, 0b011), 2, 0, 1, False, -1),
        ((0b100, 0b000), 2, 0, 1, False, 1),
        ((0b110, 0b001), 0, 2, 2, True, 1),
    ],
)
def test_check_fermionic_sign(
    state, site_from, site_to, spin, is_boundary_hop, expected
):
    """Test the calculation of the fermionic sign."""
    assert (
        check_fermionic_sign(
            state, site_from, site_to, spin, is_boundary_hop=is_boundary_hop
        )
        == expected
    )


def test_hamiltonian_dimension():
    """Test the dimension of the Hamiltonian matrix."""
    N_up, N_down, L = 2, 1, 3
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    dim = len(basis)
    assert dim == 9, f"Expected Hamiltonian dimension 9, got {dim}"


def test_hamiltonian_symm():
    """Test the symmetry of the Hamiltonian matrix."""
    N_up, N_down, L = 2, 1, 3
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    HH = bm_create_base_hamiltonian(basis, index_map, L, 1.0, 2.0, True).toarray()
    assert np.allclose(HH, HH.T), "Hamiltonian matrix is not symmetric"


@pytest.mark.parametrize(
    "N_up, N_down, L, target_states, expected",
    [
        (
            2,
            0,
            3,
            [(0b011, 0b000), (0b101, 0b000), (0b110, 0b000)],
            np.array(
                [
                    [0, -1, 1],
                    [-1, 0, -1],
                    [1, -1, 0],
                ]
            ),
        ),
        (
            0,
            2,
            3,
            [(0b000, 0b011), (0b000, 0b101), (0b000, 0b110)],
            np.array(
                [
                    [0, -1, 1],
                    [-1, 0, -1],
                    [1, -1, 0],
                ]
            ),
        ),
    ],
)
def test_create_hamiltonian_PBC(N_up, N_down, L, target_states, expected):
    """Test the construction of the Hamiltonian matrix."""
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    HH = bm_create_base_hamiltonian(basis, index_map, L, 1.0, 2.0, True).toarray()

    indices = [index_map[state] for state in target_states]
    HH_sub = HH[np.ix_(indices, indices)]

    np.testing.assert_array_equal(HH_sub, expected)


@pytest.mark.parametrize(
    "N_up, N_down, L, target_states, expected",
    [
        (
            2,
            0,
            3,
            [(0b011, 0b000), (0b101, 0b000), (0b110, 0b000)],
            np.array(
                [
                    [0, -1, 0],
                    [-1, 0, -1],
                    [0, -1, 0],
                ]
            ),
        ),
        (
            0,
            2,
            3,
            [(0b000, 0b011), (0b000, 0b101), (0b000, 0b110)],
            np.array(
                [
                    [0, -1, 0],
                    [-1, 0, -1],
                    [0, -1, 0],
                ]
            ),
        ),
    ],
)
def test_create_hamiltonian_OBC(N_up, N_down, L, target_states, expected):
    """Test the construction of the Hamiltonian matrix with open boundary conditions."""
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    HH = bm_create_base_hamiltonian(basis, index_map, L, 1.0, 2.0, PBC=False).toarray()
    _, index_map = bitmap_basis_states(L, N_up, N_down)

    indices = [index_map[state] for state in target_states]
    HH_sub = HH[np.ix_(indices, indices)]

    np.testing.assert_array_equal(HH_sub, expected)


def test_gauge_symmetry():
    """Test that the Hamiltonian is Hermitian for both gauge choices."""
    N_up, N_down, L = 2, 2, 4

    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    HH_velocity = bm_create_base_hamiltonian(
        basis, index_map, L, 1.0, 2.0, PBC=False
    ).toarray()
    HH_length = bm_create_base_hamiltonian(
        basis, index_map, L, 1.0, 2.0, PBC=False
    ).toarray()

    assert np.allclose(
        HH_velocity, HH_velocity.T.conj()
    ), "Velocity gauge Hamiltonian is not Hermitian"
    assert np.allclose(
        HH_length, HH_length.T.conj()
    ), "Length gauge Hamiltonian is not Hermitian"


def test_time_evolution_gauge():
    """Test if both gauge choices give the same doublon expectation after time evolution under the same field."""
    system = setup_hubbard_system(
        N_up=2, N_down=2, L=4, t=1.0, U=2.0, PBC=False, e_charge=1, hbar_val=1
    )

    psi_0 = bm_get_eigenstates(
        system["H_int"] + system["H_hop"] + system["H_hop"].T.conj(), num_evals=1
    )[1][:, 0]

    omega = 2 * np.pi
    vel_field = lambda t: (1 - np.cos(omega * t))
    length_field = lambda t: (-omega * np.sin(omega * t))

    psi_velocity = time_evolve_state(
        psi_0,
        system,
        gauge_choice="velocity",
        gauge_field=vel_field,
        t0=0,
        tf=1,
        num_points=300,
    )
    psi_length = time_evolve_state(
        psi_0,
        system,
        gauge_choice="length",
        gauge_field=length_field,
        t0=0,
        tf=1,
        num_points=300,
    )

    assert np.allclose(
        np.linalg.norm(psi_velocity), 1.0, atol=1e-6
    ), "State norm is not preserved in velocity gauge time evolution"
    assert np.allclose(
        np.linalg.norm(psi_length), 1.0, atol=1e-6
    ), "State norm is not preserved in length gauge time evolution"

    doublon_vel_exp = get_doublon_expectation(
        psi_velocity, system["basis"], system["Params"]["L"]
    )
    doublon_len_exp = get_doublon_expectation(
        psi_length, system["basis"], system["Params"]["L"]
    )

    assert np.allclose(
        doublon_len_exp, doublon_vel_exp, atol=1e-5
    ), "Total doublon expectation values differ between gauges after time evolution"
