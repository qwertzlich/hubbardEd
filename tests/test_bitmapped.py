"""Testing routines for the bitmapped Hamiltonian."""

import os
import sys
import numpy as np
import pytest

directory = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(directory, "..", "src")))

from hubbardEd.bitmapped import (
    bm_hopping_operator,
    check_fermionic_sign,
    bm_create_hamiltonian,
    bit_flip,
    check_bit,
)
from hubbardEd.bitmapped.basis import bitmap_basis_states


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
    "state, site_from, site_to, spin, PBC, expected",
    [
        ((0b10, 0b00), 1, 0, 1, False, 1),
        ((0b110, 0b011), 2, 0, 1, False, -1),
        ((0b100, 0b000), 2, 0, 1, False, 1),
        ((0b110, 0b001), 0, 2, 2, True, 1),
    ],
)
def test_check_fermionic_sign(state, site_from, site_to, spin, PBC, expected):
    """Test the calculation of the fermionic sign."""
    assert check_fermionic_sign(state, site_from, site_to, spin, PBC) == expected


def test_hamiltonian_dimension():
    """Test the dimension of the Hamiltonian matrix."""
    N_up, N_down, L = 2, 1, 3
    basis, index_map = bitmap_basis_states(L, N_up, N_down)
    dim = len(basis)
    assert dim == 9, f"Expected Hamiltonian dimension 9, got {dim}"


def test_hamiltonian_symm():
    """Test the symmetry of the Hamiltonian matrix."""
    N_up, N_down, L = 2, 1, 3
    HH = bm_create_hamiltonian(N_up, N_down, L, 1.0, 2.0).toarray()
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
def test_create_hamiltonian(N_up, N_down, L, target_states, expected):
    """Test the construction of the Hamiltonian matrix."""
    t, U = 1.0, 2.0
    HH = bm_create_hamiltonian(N_up, N_down, L, t, U).toarray()
    _, index_map = bitmap_basis_states(L, N_up, N_down)

    indices = [index_map[state] for state in target_states]
    HH_sub = HH[np.ix_(indices, indices)]

    np.testing.assert_array_equal(HH_sub, expected)
