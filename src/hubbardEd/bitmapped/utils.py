"""Utility functions for bitmapped basis states in the Hubbard model."""


def bit_flip(bin_state: int, site: int) -> int:
    """Flip the bit at the specified site in the given binary state."""
    return bin_state ^ (1 << site)


def check_bit(bin_state: int, site: int) -> bool:
    """Check if the bit at the specified site is set (1) or not (0)."""
    return (bin_state >> site) & 1 == 1
