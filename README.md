# HubbardEd

HubbardEd is a Python project for exact diagonalization of the 1D Fermi-Hubbard model on small lattices.  
It includes two implementations of the same physics:

- an analog, occupation-value representation (simple and explicit), and
- a bitmapped representation (more efficient and scalable for fixed spin sectors).

The code is useful for studying strongly correlated lattice electrons in finite systems, validating small analytical results, and prototyping observables and driven dynamics before moving to larger-scale methods.

## Physical model

The implemented Hamiltonian is the standard single-band 1D Hubbard model:

$$
H = -t \sum_{\langle i,j \rangle,\sigma} \left(c^\dagger_{i\sigma} c_{j\sigma} + h.c.\right) + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

with:

- $t$: nearest-neighbor hopping amplitude,
- $U$: on-site interaction strength,
- $\sigma \in \{\uparrow, \downarrow\}$: spin index.

Both periodic boundary conditions (PBC) and open boundary conditions (OBC) are supported in the bitmapped backend.

## What this project is useful for

- Exploring few-site Hubbard physics in a transparent way.
- Ground-state and low-energy spectrum calculations.
- Testing fermionic sign conventions and basis encodings.
- Computing simple correlation observables (doublons, long-range charge correlation).
- Studying time evolution under external fields in velocity and length gauges.
- Building a baseline reference for future tensor-network or mean-field implementations.

## Project structure

```
src/hubbardEd/
	analog/
		basis.py        # Occupation-based basis generation
		hamiltonian.py  # Operator construction and sparse Hamiltonian
	bitmapped/
		basis.py        # Bit-encoded spin-up/down basis states
		hamiltonian.py  # Sparse hopping + interaction matrices
		observables.py  # Doublon and long-range correlation expectation values
		time_prop.py    # Time evolution and gauge-dependent dynamics
		utils.py        # Bit utilities

tests/
	test_analog.py
	test_bitmapped.py
```

## How it works

The computational flow is:

1. Build a many-body basis in a fixed particle sector.
2. Construct sparse Hamiltonian terms:
	 - interaction part (diagonal in basis),
	 - hopping part with fermionic sign handling.
3. Form the full Hamiltonian and diagonalize low-energy states.
4. Evaluate observables from eigenvectors or propagated states.
5. (Optional) propagate in time under a gauge field.

### 1) Basis construction

- Analog backend (`analog/basis.py`):
	- encodes each site with values `0` (empty), `1` (up), `2` (down), `3` (double occupancy),
	- filters product states by total electron number `N`.

- Bitmapped backend (`bitmapped/basis.py`):
	- stores each basis state as `(up_bits, down_bits)`,
	- fixes `N_up` and `N_down` independently,
	- Hilbert dimension is
		$$
		\dim = \binom{L}{N_{\uparrow}}\binom{L}{N_{\downarrow}}.
		$$

### 2) Hamiltonian assembly

- Interaction term: counts doublons and adds `U * N_doublon` on the diagonal.
- Hopping term: applies nearest-neighbor hops and multiplies by the correct fermionic sign.
- In the bitmapped backend, right hops are constructed and then Hermitian-conjugated to include left hops.

### 3) Eigenstates

- `bm_get_eigenstates` computes low-lying eigenpairs using sparse Lanczos (`scipy.sparse.linalg.eigsh`).

### 4) Observables

- `get_doublon_expectation`: site-resolved
	$$\langle n_{i\uparrow} n_{i\downarrow} \rangle$$
	from a state vector.
- `get_LRC_expectation`: long-range charge correlation between site `0` and site `L/2`.

### 5) Time propagation

`bitmapped/time_prop.py` provides:

- `setup_hubbard_system(...)` to precompute operators,
- `time_evolve_state(...)` to propagate using midpoint sampling and
	`scipy.sparse.linalg.expm_multiply`.

Supported gauges:

- `velocity`: complex Peierls phase on hopping terms,
- `length`: scalar potential from position operator (only for OBC).


## Testing and validation

The test suite verifies core physics and numerics in both backends, including:

- basis dimensions,
- creation/annihilation and hopping operators,
- fermionic sign rules,
- Hermiticity/symmetry of Hamiltonians,
- known small-system matrix blocks and eigenvalues,
- norm preservation and gauge consistency in time propagation.

Run tests with:

```bash
python3 -m pytest 
```
