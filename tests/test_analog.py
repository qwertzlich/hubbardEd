"""Routines for testing the functions in hubbard_solver."""

import os
import sys
import numpy as np
import pytest

directory = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(directory, "..", "src")))

from hubbardEd.analog import (
    basis_states,
    creation_operator,
    annihilation_operator,
    check_fermionic_sign,
    hamiltonian,
)


@pytest.mark.parametrize("N, L, expected_len", [(2, 2, 6), (2, 1, 1), (3, 2, 4)])
def test_basis_states_dim(N, L, expected_len):
    """Test the length of the basis states."""
    basis, _ = basis_states(N, L)
    assert len(basis) == expected_len


@pytest.mark.parametrize(
    "state, site, spin, expected",
    [
        (np.array([0, 1, 0, 0]), 0, 1, np.array([1, 1, 0, 0])),
        (np.array([0, 1, 0, 0]), 0, 2, np.array([2, 1, 0, 0])),
        (np.array([2, 0, 1, 3]), 0, 2, None),
        (np.array([2, 0, 1, 3]), 3, 1, None),
    ],
)
def test_creation_operator(state, site, spin, expected):
    """Test the creation operator."""
    if expected is None:
        assert creation_operator(state, site, spin) is None
    else:
        np.testing.assert_array_equal(creation_operator(state, site, spin), expected)


@pytest.mark.parametrize(
    "state, site, spin, expected",
    [
        (np.array([0, 1, 0, 0]), 1, 1, np.array([0, 0, 0, 0])),
        (np.array([0, 1, 0, 3]), 3, 2, np.array([0, 1, 0, 1])),
        (np.array([0, 0, 1, 3]), 0, 2, None),
        (np.array([2, 0, 1, 3]), 0, 1, None),
    ],
)
def test_annihilation_operator(state, site, spin, expected):
    """Test the annihilation operator."""
    if expected is None:
        assert annihilation_operator(state, site, spin) is None
    else:
        np.testing.assert_array_equal(
            annihilation_operator(state, site, spin), expected
        )


@pytest.mark.parametrize(
    "state, site_from, site_to, spin, expected",
    [
        (np.array([3, 0]), 0, 1, 1, -1),
        (np.array([3, 0]), 0, 1, 2, 1),
        (np.array([0, 3]), 1, 0, 1, 1),
        (np.array([0, 3]), 1, 0, 2, -1),
    ],
)
def test_check_fermionic_sign(state, site_from, site_to, spin, expected):
    """Test the fermionic sign calculation."""
    assert check_fermionic_sign(state, site_from, site_to, spin) == expected


def test_hamiltonian_dim():
    """Test the Hamiltonian construction."""
    N, L, t, U = 2, 2, 1.0, 2.0
    HH = hamiltonian(N, L, t, U)
    assert HH.shape == (6, 6)


def test_hamiltonian_simple_case():
    """Test the Hamiltonian construction for L=2, N=2 with eigenvalue verification."""
    N, L, t, U = 2, 2, 1.0, 2.0

    HH = hamiltonian(
        N, L, t, U
    ).toarray()  # Convert sparse matrix to dense for testing since it is low-dimensional
    _, index_map = basis_states(N, L)

    target_states = [(1, 2), (2, 1), (3, 0), (0, 3)]
    indices = [index_map[state] for state in target_states]

    HH_sub = HH[np.ix_(indices, indices)]

    expected_eigenvalues = np.sort(
        [
            0.0,
            U,
            0.5 * (U + np.sqrt(U**2 + 16 * t**2)),
            0.5 * (U - np.sqrt(U**2 + 16 * t**2)),
        ]
    )

    actual_eigenvalues = np.sort(np.linalg.eigvalsh(HH_sub))

    np.testing.assert_array_almost_equal(
        actual_eigenvalues,
        expected_eigenvalues,
        decimal=6,
        err_msg="Die Eigenwerte (Energien) stimmen nicht!",
    )

    np.testing.assert_array_almost_equal(
        HH_sub, HH_sub.T, decimal=10, err_msg="Die Matrix ist nicht symmetrisch!"
    )

    triplett_states = [(1, 1), (2, 2)]
    for s in triplett_states:
        idx = index_map[s]
        row_sum_offdiag = np.sum(np.abs(HH[idx, :])) - np.abs(HH[idx, idx])
        assert (
            row_sum_offdiag == 0
        ), f"Zustand {s} koppelt fälschlicherweise mit anderen Zuständen!"
