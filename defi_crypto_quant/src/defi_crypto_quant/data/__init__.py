"""Data loaders and synthetic market generators for DeFi & Crypto Quant."""

from .loader import (
    generate_synthetic_dex_pools,
    generate_synthetic_funding_rates,
    generate_synthetic_mempool_swaps,
)

__all__ = [
    "generate_synthetic_dex_pools",
    "generate_synthetic_funding_rates",
    "generate_synthetic_mempool_swaps",
]
