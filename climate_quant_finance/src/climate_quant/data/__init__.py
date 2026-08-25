"""Data loader & synthetic generator for Climate Quant Finance."""

from .loader import (
    generate_corporate_climate_universe,
    generate_satellite_plume_observations,
    generate_climate_equity_prices,
    generate_disclosed_emissions_universe,
)

__all__ = [
    "generate_corporate_climate_universe",
    "generate_satellite_plume_observations",
    "generate_climate_equity_prices",
    "generate_disclosed_emissions_universe",
]
