"""Unit tests for the cavity-coupled Hubbard model Hamiltonian construction."""

import numpy as np
import pytest

from hubbardEd.cavity import (
    spinless_fermion_basis,
    create_cavity_nn_int_matrix,
    create_cavity_fermion_hopping_matrix,
    create_cavity_hamiltonian,
)


@pytest.mark.parametrize(
    "num_sites, num_fermions, expected_len",
    [
        (4, 2, 6),
        (6, 3, 20),
        (2, 1, 2),
    ],
)
def test_fermion_basis(num_sites, num_fermions, expected_len):
    """Test the generation of the spinless fermion basis."""
    basis, index_map = spinless_fermion_basis(num_sites, num_fermions)
    assert (
        len(basis) == expected_len
    ), f"Expected {expected_len} basis states, got {len(basis)}"


def test_cavity_nn_int_matrix():
    """Test the construction of the nearest-neighbor interaction matrix."""
    num_sites = 2
    num_fermions = 1
    U = 1.0
    basis, index_map = spinless_fermion_basis(num_sites, num_fermions)
    nn_int_matrix = create_cavity_nn_int_matrix(basis, index_map, num_sites, U)
    assert nn_int_matrix.shape == (2, 2), "Interaction matrix has incorrect shape"
    assert np.allclose(
        nn_int_matrix.diagonal(), [-0.25, -0.25]
    ), "Diagonal elements of interaction matrix are incorrect"


def test_cavity_fermion_hopping_matrix():
    """Test the construction of the fermion hopping matrix."""
    num_sites = 2
    num_fermions = 1
    t = 1.0
    basis, index_map = spinless_fermion_basis(num_sites, num_fermions)
    hopping_matrix = create_cavity_fermion_hopping_matrix(
        basis, index_map, num_sites, t, num_fermions
    )
    hopping_matrix += hopping_matrix.T.conj()  # Convert to dense for testing
    assert hopping_matrix.shape == (2, 2), "Hopping matrix has incorrect shape"
    assert np.isclose(
        hopping_matrix[0, 1], -t
    ), "Off-diagonal element of hopping matrix is incorrect"
    assert np.isclose(
        hopping_matrix[1, 0], -t
    ), "Off-diagonal element of hopping matrix is incorrect"


def test_create_cavity_hamiltonian():
    """Test the construction of the full cavity-coupled Hamiltonian."""
    num_sites = 2
    num_fermions = 1
    t = 1.0
    U = 1.0
    N_photons = 1
    g = 0.5
    Omega = 1.0

    basis, index_map = spinless_fermion_basis(num_sites, num_fermions)
    HH = create_cavity_hamiltonian(
        basis, index_map, num_sites, t, U, N_photons, g, Omega
    ).toarray()  # Test case is low dimensional, so we can convert to dense for testing
    # Test shape of the Hamiltonian
    expected_dim = (N_photons + 1) * len(basis)
    assert HH.shape == (expected_dim, expected_dim), "Hamiltonian has incorrect shape"
    # Test that the Hamiltonian is Hermitian
    assert np.allclose(HH, HH.conj().T), "Hamiltonian is not Hermitian"
    # check Eigenvalues match expected values
    L = num_sites
    expected_matrix = np.array(
        [
            [-U / 4, -t * np.cos(g / np.sqrt(L)), 0, 1j * t * np.sin(g / np.sqrt(L))],
            [-t * np.cos(g / np.sqrt(L)), -U / 4, -1j * t * np.sin(g / np.sqrt(L)), 0],
            [
                0,
                1j * t * np.sin(g / np.sqrt(L)),
                Omega - U / 4,
                -t * np.cos(g / np.sqrt(L)),
            ],
            [
                -1j * t * np.sin(g / np.sqrt(L)),
                0,
                -t * np.cos(g / np.sqrt(L)),
                Omega - U / 4,
            ],
        ]
    )
    assert np.allclose(
        expected_matrix, expected_matrix.conj().T
    ), "Expected Hamiltonian matrix is not Hermitian"
    expected_eigenvalues = np.linalg.eigvalsh(expected_matrix)
    computed_eigenvalues = np.linalg.eigvalsh(HH)
    assert np.allclose(
        computed_eigenvalues, expected_eigenvalues
    ), "Eigenvalues of the Hamiltonian do not match expected values"
