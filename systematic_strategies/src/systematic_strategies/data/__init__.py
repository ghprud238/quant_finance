"""Data layer for systematic strategies."""

from .synthetic import (
    generate_date_range,
    generate_equities_data,
    generate_pairs_data,
    generate_macro_data,
    generate_cross_sectional_universe,
    generate_and_save_all_sample_data,
)
from .loader import (
    load_equities,
    load_pairs,
    load_macro,
    load_cross_sectional,
)

__all__ = [
    "generate_date_range",
    "generate_equities_data",
    "generate_pairs_data",
    "generate_macro_data",
    "generate_cross_sectional_universe",
    "generate_and_save_all_sample_data",
    "load_equities",
    "load_pairs",
    "load_macro",
    "load_cross_sectional",
]
