# HubbardEd

This project was developed with assistance from AI tools during documentation and code cleanup.

HubbardEd is a Python project for exact diagonalization of Hubbard-type models on small lattices. It provides implementations of two different physics models, each with one or more computational backends for evaluating ground states, low-energy spectra, observables, and time-evolved dynamics.

## Physical models

### Model 1: Spin-½ Fermi-Hubbard model (Analog & Bitmapped backends)

The standard 1D Fermi-Hubbard model with repulsive on-site interaction:

$$
H = -t \sum_{\langle i,j \rangle,\sigma} \left(c^\dagger_{i\sigma} c_{j\sigma} + h.c.\right) + U \sum_i n_{i\uparrow} n_{i\downarrow}
$$

where:
- $t$: nearest-neighbor hopping amplitude,
- $U$: on-site interaction strength,
- $\sigma \in \{\uparrow, \downarrow\}$: spin index,
- $n_{i\sigma} = c^\dagger_{i\sigma} c_{i\sigma}$: occupation number.

Both periodic boundary conditions (PBC) and open boundary conditions (OBC) are supported.

**Use cases:**
- Exploring few-site Hubbard physics in a transparent way.
- Ground-state and low-energy spectrum calculations.
- Testing fermionic sign conventions and basis encodings.
- Computing correlation observables (doublons, long-range particle correlations).
- Studying time evolution under external fields.
- Validating tensor-network or mean-field implementations.

### Model 2: Spinless fermions with nearest-neighbor interactions coupled to cavity modes (Cavity backend)

A model of spinless fermions with nearest-neighbor interactions coupled to a quantized cavity mode via the quantized Peierls substitution in Coulomb gauge:

$$
\begin{aligned}
H &= -t_h \sum_{j=1}^L \left(e^{i g/\sqrt{L}(a + a^\dagger)} c_j^\dagger c_{j+1} + h.c.\right) +\\ 
&U \sum_{j=1}^{L} \left(n_j - \frac{1}{2}\right)\left(n_{j+1} - \frac{1}{2}\right) + \Omega a^\dagger a
\end{aligned}
$$

where:
- $L$: number of lattice sites,
- $t_h$: hopping amplitude,
- $U$: nearest-neighbor interaction strength,
- $g$: light-matter coupling strength,
- $\Omega$: cavity photon frequency,
- $a^\dagger, a$: photon creation and annihilation operators,
- $n_j = c_j^\dagger c_j$: fermion number operator at site $j$.

The full Hilbert space is the tensor product of the photonic (Fock) and electronic spaces: $|n_{\text{ph}}\rangle \otimes |n_{\text{el}}\rangle$.

**Use cases:**
- Studying light-matter interactions in strongly correlated systems.
- Exploring polariton physics in cavity-coupled fermionic chains.
- Computing correlation functions in coupled electron-photon systems.

## Project structure

```
src/hubbardEd/
	analog/
		basis.py        # Occupation-based basis generation (Spin-½ Hubbard)
		hamiltonian.py  # Operator construction and sparse Hamiltonian
	bitmapped/
		basis.py        # Bit-encoded spin-up/down basis states (Spin-½ Hubbard)
		hamiltonian.py  # Sparse hopping + on-site interaction matrices
		observables.py  # Doublon and long-range correlation expectation values
		time_prop.py    # Time evolution under external gauge fields
		utils.py        # Bit utilities
	cavity/
		basis_cav.py        # Spinless fermion basis generation
		hamiltonian_cav.py  # Cavity-coupled Hamiltonian with quantized Peierls substitution
		observables_cav.py  # Long-range charge correlation observable

tests/
	test_analog.py
	test_bitmapped.py
	test_cavity.py
```

## Implementation details

### Basis construction

**Analog backend** (`analog/basis.py`):
- Encodes each site with occupation values: `0` (empty), `1` (spin-up), `2` (spin-down), `3` (double occupancy).
- Generates product states and filters by total electron number $N$.

**Bitmapped backend** (`bitmapped/basis.py`):
- Represents each basis state as $(n_{\uparrow}, n_{\downarrow})$ where each is a bitmask with one bit per site.
- Fixes $N_{\uparrow}$ and $N_{\downarrow}$ independently.
- Hilbert space dimension: $\dim = \binom{L}{N_{\uparrow}} \binom{L}{N_{\downarrow}}$.

**Cavity backend** (`cavity/basis_cav.py`):
- Generates spinless fermion basis states via bitmapped representation (fixed fermion number $N_f$).
- Photonic basis (Fock states from 0 to $N_{\text{photons}}$) is constructed during Hamiltonian assembly.
- Full Hilbert space is the tensor product: $\dim = \binom{L}{N_f} \times (N_{\text{photons}} + 1)$.

### Hamiltonian assembly

**Spin-½ Hubbard backends** (Analog & Bitmapped):
- **On-site interaction:** Counts double occupancy and adds $U \times (\text{doublon count})$ to diagonal.
- **Hopping term:** Applies nearest-neighbor hops $c^\dagger_{i\sigma} c_{j\sigma} + h.c.$ with correct fermionic sign.
  - Analog backend: Uses explicit anticommutation rules via `check_fermionic_sign`.
  - Bitmapped backend: Constructs right-hopping only, then adds Hermitian conjugate.

**Cavity backend**:
- **Nearest-neighbor interaction:** Uses particle-hole-symmetric formulation with factors $(n_i - \frac{1}{2})(n_{i+1} - \frac{1}{2})$.
- **Hopping term:** Spinless fermion hopping with fermionic signs.
- **Cavity coupling:** Uses exponential displacement operators $e^{\pm i g/\sqrt{L}(a + a^\dagger)}$ applied to hopping terms.
- **Photon frequency:** Contributes diagonal energy $\Omega \times n_{\text{photons}}$.

### Eigenstate computation

Both backends use sparse Lanczos diagonalization (`scipy.sparse.linalg.eigsh`) to compute low-lying eigenpairs.

### Observables

**Spin-½ Hubbard backends** (`bitmapped/observables.py`):
- **Doublon expectation:** Site-resolved $\langle n_{i\uparrow} n_{i\downarrow} \rangle$.
- **Long-range charge correlation:** $\langle n_0 n_{L/2} \rangle$ (between site 0 and middle site).

**Cavity backend** (`cavity/observables_cav.py`):
- **Site correlation:** Long-range charge correlation $\langle n_i n_j \rangle$ between arbitrary sites $i$ and $j$, computed in the full electron-photon Hilbert space.

### Time propagation

**Bitmapped backend** (`bitmapped/time_prop.py`):
- Provides Hamiltonian setup and time evolution under velocity or length gauge fields.
- Uses midpoint sampling and matrix exponential propagation (`scipy.sparse.linalg.expm_multiply`).

**Supported gauges:**
- `velocity`: Complex Peierls phase on hopping terms.
- `length`: Scalar potential from position operator (OBC only).

## Testing and validation

The test suite verifies core physics and numerics:

- Basis dimensions and orthonormality.
- Creation/annihilation and hopping operator correctness.
- Fermionic sign rules and anticommutation relations.
- Hermiticity and symmetry of Hamiltonians.
- Known small-system matrix elements and eigenvalues.
- Norm preservation and gauge consistency in time propagation.

Run tests with:

```bash
python3 -m pytest
```

