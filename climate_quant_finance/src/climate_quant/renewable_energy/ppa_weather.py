"""Renewable Energy PPA Valuation & Weather Derivatives Engine (Project 40).

Implements:
1. Hourly Solar & Wind Generation Simulation (Weibull Copula & Diurnal Solar Geometry).
2. Hourly Electricity Spot Price Modeling with Merit-Order Price Suppression (The Duck Curve & Cannibalization).
3. Pay-As-Produced (PAP) vs. Baseload (Fixed Volume) PPA Valuation & Shaping Risk.
4. Weather Derivatives Pricing (HDD/CDD Swaps, Calls, Puts, Collars via Ornstein-Uhlenbeck & Burn Analysis).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# PART 1: RENEWABLE ENERGY PPA VALUATION & CANNIBALIZATION ENGINE
# =============================================================================

@dataclass
class RenewableAssetConfig:
    """Configuration specifications for a renewable generation asset."""
    asset_type: str = "solar"                 # 'solar' or 'wind'
    capacity_mw: float = 100.0                 # Installed capacity in MW
    latitude_deg: float = 35.0                 # Latitude for solar geometry
    weibull_k: float = 2.1                     # Weibull shape parameter for wind
    weibull_c: float = 8.8                     # Weibull scale parameter (m/s)
    cut_in_speed: float = 3.0                  # Wind cut-in speed (m/s)
    rated_speed: float = 12.0                  # Wind rated speed (m/s)
    cut_out_speed: float = 25.0                # Wind cut-out speed (m/s)
    performance_ratio: float = 0.82            # Solar performance ratio (derating)
    annual_degradation_rate: float = 0.005     # 0.5% annual efficiency degradation
    opex_per_mwh: float = 7.50                 # Operating cost per MWh ($/MWh)
    balancing_cost_per_mwh: float = 2.50       # Imbalance & grid balancing fee ($/MWh)
    discount_rate: float = 0.065               # Cost of Capital / WACC (6.5%)


@dataclass
class HourlyGenerationProfile:
    """Hourly renewable generation timeseries and yield statistics."""
    timestamps: pd.DatetimeIndex
    generation_mwh: np.ndarray
    capacity_factor: float
    total_annual_generation_mwh: float
    peak_generation_mw: float
    profile_series: pd.Series = field(init=False)

    def __post_init__(self):
        self.profile_series = pd.Series(self.generation_mwh, index=self.timestamps, name="Generation_MWh")


class PPAType(str, Enum):
    """PPA Contract Offtake Structures."""
    PAY_AS_PRODUCED = "PAY_AS_PRODUCED"       # Offtaker pays fixed strike for actual generation Q_t
    BASELOAD = "BASELOAD"                     # Generator delivers flat hourly volume Q_base
    MERCHANT = "MERCHANT"                     # 100% merchant spot exposure
    COLLAR_PPA = "COLLAR_PPA"                 # Floor and Cap on market price for generated volume


@dataclass
class PPAContract:
    """PPA Contract terms and settlement parameters."""
    contract_type: PPAType = PPAType.PAY_AS_PRODUCED
    strike_price: float = 55.0                 # Fixed PPA price ($/MWh)
    fixed_volume_mw: Optional[float] = None    # Flat volume per hour for BASELOAD PPA
    floor_price: Optional[float] = None        # Price floor for COLLAR PPA ($/MWh)
    cap_price: Optional[float] = None          # Price cap for COLLAR PPA ($/MWh)
    tenor_years: int = 10                      # Contract duration in years
    settlement_frequency: str = "hourly"      # Settlement granularity


@dataclass
class PPAValuationResult:
    """Valuation metrics, cash flows, and cannibalization diagnostics for a PPA."""
    contract_type: str
    strike_price: float
    npv_revenue_usd: float
    npv_net_cashflow_usd: float
    npv_per_kw_installed: float
    mean_baseload_price: float
    capture_price: float
    capture_rate: float
    cannibalization_discount_pct: float
    shaping_cost_total_usd: float
    balancing_cost_total_usd: float
    annual_generation_mwh: float
    annual_gross_revenue_usd: float
    annual_net_cashflow_usd: float
    revenue_volatility_pct: float
    var_95_annual_revenue_usd: float
    summary_metrics: Dict[str, Any]

    def summary_table(self) -> pd.DataFrame:
        """Formats results as a clean diagnostic DataFrame."""
        records = [
            ("PPA Structure", self.contract_type),
            ("PPA Strike Price", f"${self.strike_price:.2f} / MWh"),
            ("NPV Net Cash Flows (10Y)", f"${self.npv_net_cashflow_usd:,.0f}"),
            ("NPV per kW Installed", f"${self.npv_per_kw_installed:,.2f} / kW"),
            ("Mean Market Baseload Price", f"${self.mean_baseload_price:.2f} / MWh"),
            ("Renewable Capture Price", f"${self.capture_price:.2f} / MWh"),
            ("Capture Rate (vs Baseload)", f"{self.capture_rate:.2%}"),
            ("Cannibalization Discount", f"{self.cannibalization_discount_pct:.2f}%"),
            ("Total Shaping Cost (10Y)", f"${self.shaping_cost_total_usd:,.0f}"),
            ("Total Balancing Cost (10Y)", f"${self.balancing_cost_total_usd:,.0f}"),
            ("Annual Generation (P50)", f"{self.annual_generation_mwh:,.0f} MWh"),
            ("Annual Net Cashflow", f"${self.annual_net_cashflow_usd:,.0f} / year"),
            ("Annual Revenue Volatility", f"{self.revenue_volatility_pct:.2f}%"),
            ("95% Value at Risk (Revenue)", f"${self.var_95_annual_revenue_usd:,.0f}"),
        ]
        return pd.DataFrame(records, columns=["Metric", "Value"])


class RenewablePPAValuator:
    """Comprehensive Valuation Engine for Wind and Solar Power Purchase Agreements."""

    def __init__(self, config: Optional[RenewableAssetConfig] = None):
        self.config = config or RenewableAssetConfig()

    def simulate_hourly_solar_profile(
        self,
        year: int = 2026,
        seed: int = 42,
    ) -> HourlyGenerationProfile:
        """Simulates 8760 hourly solar generation yields using solar geometry and cloud stochasticity."""
        np.random.seed(seed)
        start_ts = pd.Timestamp(f"{year}-01-01 00:00:00")
        end_ts = pd.Timestamp(f"{year}-12-31 23:00:00")
        timestamps = pd.date_range(start_ts, end_ts, freq="h")
        n_hours = len(timestamps)

        day_of_year = timestamps.dayofyear.values
        hour_of_day = timestamps.hour.values

        # Solar declination angle: delta = 23.45 * sin(2pi/365 * (284 + day))
        delta_rad = np.radians(23.45 * np.sin(2.0 * np.pi / 365.0 * (284.0 + day_of_year)))
        lat_rad = np.radians(self.config.latitude_deg)

        # Solar hour angle: omega = 15 * (hour - 12) degrees
        omega_rad = np.radians(15.0 * (hour_of_day + 0.5 - 12.0))

        # Solar elevation angle: sin(alpha) = sin(lat)*sin(delta) + cos(lat)*cos(delta)*cos(omega)
        sin_elev = np.sin(lat_rad) * np.sin(delta_rad) + np.cos(lat_rad) * np.cos(delta_rad) * np.cos(omega_rad)
        sin_elev = np.maximum(0.0, sin_elev)

        # Clear sky Global Horizontal Irradiance (GHI) W/m^2 (peak approx 1050 W/m^2)
        ghi_clearsky = 1050.0 * sin_elev ** 1.15

        # Stochastic cloud clearness index (AR(1) process with Beta distribution)
        cloud_factor = np.zeros(n_hours)
        cloud_factor[0] = 0.85
        ar_phi = 0.88
        for t in range(1, n_hours):
            shock = np.random.beta(5.0, 1.5)
            cloud_factor[t] = ar_phi * cloud_factor[t-1] + (1.0 - ar_phi) * shock

        ghi_actual = ghi_clearsky * np.clip(cloud_factor, 0.15, 1.0)

        # Solar Generation: Q_t = Capacity_MW * (GHI / 1000 W/m^2) * Performance_Ratio
        generation_mwh = self.config.capacity_mw * (ghi_actual / 1000.0) * self.config.performance_ratio
        generation_mwh = np.maximum(0.0, generation_mwh)

        total_annual = float(np.sum(generation_mwh))
        cf = total_annual / (self.config.capacity_mw * n_hours)

        return HourlyGenerationProfile(
            timestamps=timestamps,
            generation_mwh=generation_mwh,
            capacity_factor=cf,
            total_annual_generation_mwh=total_annual,
            peak_generation_mw=float(np.max(generation_mwh)),
        )

    def simulate_hourly_wind_profile(
        self,
        year: int = 2026,
        seed: int = 42,
    ) -> HourlyGenerationProfile:
        """Simulates 8760 hourly wind power yields via Weibull Copula and turbine power curves."""
        np.random.seed(seed)
        start_ts = pd.Timestamp(f"{year}-01-01 00:00:00")
        end_ts = pd.Timestamp(f"{year}-12-31 23:00:00")
        timestamps = pd.date_range(start_ts, end_ts, freq="h")
        n_hours = len(timestamps)

        day_of_year = timestamps.dayofyear.values
        hour_of_day = timestamps.hour.values

        # Seasonal scale adjustment: stronger winds in winter/spring
        seasonal_mult = 1.0 + 0.18 * np.cos(2.0 * np.pi * (day_of_year - 20) / 365.0)
        # Diurnal cycle: nocturnal wind jet boost
        diurnal_mult = 1.0 - 0.06 * np.sin(2.0 * np.pi * (hour_of_day - 6) / 24.0)
        c_adjusted = self.config.weibull_c * seasonal_mult * diurnal_mult

        # Gaussian Copula for AR(1) temporal persistence while preserving exact Weibull marginals
        phi = 0.90
        z = np.zeros(n_hours)
        z[0] = np.random.normal(0, 1)
        for t in range(1, n_hours):
            z[t] = phi * z[t-1] + np.sqrt(1.0 - phi**2) * np.random.normal(0, 1)

        # Probability integral transform
        u = norm.cdf(z)
        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        wind_speeds = c_adjusted * (-np.log(1.0 - u)) ** (1.0 / self.config.weibull_k)

        # Power curve conversion
        v_in = self.config.cut_in_speed
        v_r = self.config.rated_speed
        v_out = self.config.cut_out_speed
        cap = self.config.capacity_mw

        gen = np.zeros(n_hours)
        for t in range(n_hours):
            v = wind_speeds[t]
            if v < v_in or v >= v_out:
                gen[t] = 0.0
            elif v_in <= v < v_r:
                gen[t] = cap * ((v - v_in) / (v_r - v_in)) ** 3.0
            else:  # v_r <= v < v_out
                gen[t] = cap

        total_annual = float(np.sum(gen))
        cf = total_annual / (self.config.capacity_mw * n_hours)

        return HourlyGenerationProfile(
            timestamps=timestamps,
            generation_mwh=gen,
            capacity_factor=cf,
            total_annual_generation_mwh=total_annual,
            peak_generation_mw=float(np.max(gen)),
        )

    def simulate_electricity_spot_prices(
        self,
        generation_profile: HourlyGenerationProfile,
        base_price_mwh: float = 60.0,
        cannibalization_beta: float = 0.35,
        volatility: float = 0.30,
        seed: int = 42,
    ) -> pd.Series:
        """Simulates hourly electricity spot prices with merit-order cannibalization (Duck Curve)."""
        np.random.seed(seed)
        timestamps = generation_profile.timestamps
        n_hours = len(timestamps)

        day_of_year = timestamps.dayofyear.values
        hour_of_day = timestamps.hour.values
        day_of_week = timestamps.dayofweek.values

        # Seasonal shape (higher in winter/summer)
        s_seasonal = 1.0 + 0.15 * np.cos(2.0 * np.pi * (day_of_year - 20) / 365.0) + 0.10 * np.cos(4.0 * np.pi * (day_of_year - 200) / 365.0)
        # Diurnal shape without solar: morning peak (8am) and evening peak (8pm)
        s_diurnal = 1.0 + 0.25 * np.sin(2.0 * np.pi * (hour_of_day - 16) / 24.0) + 0.15 * np.sin(4.0 * np.pi * (hour_of_day - 8) / 24.0)
        # Weekend demand reduction
        s_weekend = np.where(day_of_week >= 5, 0.88, 1.0)

        # Fundamental unconstrained price
        p_fundamental = base_price_mwh * s_seasonal * s_diurnal * s_weekend

        # Merit-Order Cannibalization Effect: Solar/Wind suppresses price when output is high
        q_rel = generation_profile.generation_mwh / (self.config.capacity_mw + 1e-6)
        p_cannibalized = p_fundamental - (cannibalization_beta * base_price_mwh * q_rel)

        # Autoregressive stochastic shocks & price spikes
        p_series = np.zeros(n_hours)
        p_series[0] = p_cannibalized[0]
        phi = 0.82
        sigma_hourly = volatility * base_price_mwh / np.sqrt(8760)

        for t in range(1, n_hours):
            eps = np.random.normal(0, sigma_hourly)
            spike = np.random.exponential(80.0) if np.random.random() < 0.004 else 0.0
            if q_rel[t] > 0.80 and np.random.random() < 0.02:
                neg_spike = -np.random.uniform(15.0, 40.0)
            else:
                neg_spike = 0.0

            p_series[t] = p_cannibalized[t] + phi * (p_series[t-1] - p_cannibalized[t-1]) + eps + spike + neg_spike

        return pd.Series(p_series, index=timestamps, name="Spot_Price_MWh")

    def calculate_capture_metrics(
        self,
        generation_mwh: np.ndarray,
        spot_prices: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Calculates Baseload Price, Capture Price, Capture Rate, and Cannibalization Discount."""
        mean_baseload = float(np.mean(spot_prices))
        total_gen = float(np.sum(generation_mwh))
        if total_gen > 1e-6:
            capture_price = float(np.sum(generation_mwh * spot_prices) / total_gen)
        else:
            capture_price = mean_baseload

        capture_rate = capture_price / (mean_baseload + 1e-8)
        cannibalization_discount = (1.0 - capture_rate) * 100.0

        return mean_baseload, capture_price, capture_rate, cannibalization_discount

    def value_ppa_contract(
        self,
        generation_profile: HourlyGenerationProfile,
        spot_prices_series: pd.Series,
        contract: PPAContract,
    ) -> PPAValuationResult:
        """Values a PPA contract across its multi-year tenor with degradation and risk analysis."""
        q_base = generation_profile.generation_mwh
        spot_prices = spot_prices_series.values
        tenor = contract.tenor_years
        r = self.config.discount_rate
        deg = self.config.annual_degradation_rate

        mean_base_p, cap_p, cap_rate, cann_disc = self.calculate_capture_metrics(q_base, spot_prices)

        annual_gross_rev = np.zeros(tenor)
        annual_net_cf = np.zeros(tenor)
        annual_shaping_costs = np.zeros(tenor)
        annual_balancing_costs = np.zeros(tenor)
        annual_gen_mwh = np.zeros(tenor)

        if contract.contract_type == PPAType.BASELOAD:
            if contract.fixed_volume_mw is not None:
                q_fixed_hourly = contract.fixed_volume_mw
            else:
                q_fixed_hourly = float(np.mean(q_base))
        else:
            q_fixed_hourly = 0.0

        for yr in range(tenor):
            deg_factor = (1.0 - deg) ** yr
            q_yr = q_base * deg_factor
            annual_gen_mwh[yr] = float(np.sum(q_yr))

            if contract.contract_type == PPAType.PAY_AS_PRODUCED:
                gross_rev = float(np.sum(contract.strike_price * q_yr))
                shaping_cost = 0.0
                balancing_cost = float(np.sum(self.config.balancing_cost_per_mwh * q_yr))

            elif contract.contract_type == PPAType.BASELOAD:
                fixed_rev = float(contract.strike_price * q_fixed_hourly * len(q_yr))
                spot_settlement = float(np.sum(spot_prices * (q_yr - q_fixed_hourly)))
                gross_rev = fixed_rev + spot_settlement
                pure_capture_rev = float(np.sum(spot_prices * q_yr))
                shaping_cost = max(0.0, pure_capture_rev - spot_settlement)
                balancing_cost = float(np.sum(self.config.balancing_cost_per_mwh * q_yr))

            elif contract.contract_type == PPAType.COLLAR_PPA:
                floor = contract.floor_price or 40.0
                cap = contract.cap_price or 70.0
                realized_p = np.clip(spot_prices, floor, cap)
                gross_rev = float(np.sum(realized_p * q_yr))
                shaping_cost = 0.0
                balancing_cost = float(np.sum(self.config.balancing_cost_per_mwh * q_yr))

            else:  # MERCHANT
                gross_rev = float(np.sum(spot_prices * q_yr))
                shaping_cost = 0.0
                balancing_cost = float(np.sum(self.config.balancing_cost_per_mwh * q_yr))

            opex = float(self.config.opex_per_mwh * np.sum(q_yr))
            net_cf = gross_rev - opex - balancing_cost

            annual_gross_rev[yr] = gross_rev
            annual_net_cf[yr] = net_cf
            annual_shaping_costs[yr] = shaping_cost
            annual_balancing_costs[yr] = balancing_cost

        # Discounted Cash Flows (NPV)
        discount_factors = np.array([1.0 / ((1.0 + r) ** (yr + 1)) for yr in range(tenor)])
        npv_gross = float(np.sum(annual_gross_rev * discount_factors))
        npv_net = float(np.sum(annual_net_cf * discount_factors))
        npv_per_kw = npv_net / (self.config.capacity_mw * 1000.0)

        # Revenue Risk & VaR (Simulate 500 weather/market variations)
        mc_annual_revs = np.zeros(500)
        for i in range(500):
            w_mult = np.random.normal(1.0, 0.08)
            p_mult = np.random.normal(1.0, 0.12)
            mc_annual_revs[i] = annual_gross_rev[0] * w_mult * p_mult

        rev_vol = float(np.std(mc_annual_revs) / np.mean(mc_annual_revs) * 100.0)
        var_95 = float(np.percentile(mc_annual_revs, 5.0))

        summary = {
            "NPV_Net_USD": npv_net,
            "NPV_Gross_USD": npv_gross,
            "Capture_Price": cap_p,
            "Baseload_Price": mean_base_p,
            "Capture_Rate": cap_rate,
            "Cannibalization_Pct": cann_disc,
            "Annual_P50_Generation_MWh": annual_gen_mwh[0],
            "Annual_Net_Cashflow_USD": annual_net_cf[0],
            "Capacity_Factor": generation_profile.capacity_factor,
        }

        return PPAValuationResult(
            contract_type=contract.contract_type.value,
            strike_price=contract.strike_price,
            npv_revenue_usd=npv_gross,
            npv_net_cashflow_usd=npv_net,
            npv_per_kw_installed=npv_per_kw,
            mean_baseload_price=mean_base_p,
            capture_price=cap_p,
            capture_rate=cap_rate,
            cannibalization_discount_pct=cann_disc,
            shaping_cost_total_usd=float(np.sum(annual_shaping_costs * discount_factors)),
            balancing_cost_total_usd=float(np.sum(annual_balancing_costs * discount_factors)),
            annual_generation_mwh=float(annual_gen_mwh[0]),
            annual_gross_revenue_usd=float(annual_gross_rev[0]),
            annual_net_cashflow_usd=float(annual_net_cf[0]),
            revenue_volatility_pct=rev_vol,
            var_95_annual_revenue_usd=var_95,
            summary_metrics=summary,
        )


# =============================================================================
# PART 2: WEATHER DERIVATIVES PRICING & DEGREE DAY INDICES
# =============================================================================

@dataclass
class TemperatureModelParams:
    """Parameters for Continuous Mean-Reverting Daily Temperature SDE."""
    base_temp_A: float = 14.5                  # Annual baseline temperature (°C)
    warming_trend_B: float = 0.035             # Climate warming trend (°C/year)
    seasonal_amp_C: float = 11.5               # Seasonal harmonic amplitude (°C)
    phase_phi: float = 108.0                   # Seasonal phase lag (days from Jan 1)
    kappa: float = 0.28                        # Mean-reversion speed (half-life = 2.47 days)
    sigma0: float = 2.80                       # Constant volatility (°C)
    sigma1: float = 1.10                       # Seasonal volatility amplitude (°C)
    sigma_phase: float = 45.0                  # Phase for seasonal volatility
    market_price_of_risk_lambda: float = 0.06  # Risk premium parameter (MPR)


class WeatherContractType(str, Enum):
    """Weather Derivative Index Structuring Types."""
    HDD_SWAP = "HDD_SWAP"                     # Heating Degree Day Swap
    CDD_SWAP = "CDD_SWAP"                     # Cooling Degree Day Swap
    HDD_CALL = "HDD_CALL"                     # Cap protection against severe cold winters
    HDD_PUT = "HDD_PUT"                       # Floor protection against warm winters
    CDD_CALL = "CDD_CALL"                     # Protection against extreme summer heat waves
    CDD_PUT = "CDD_PUT"                       # Protection against cool summers / low AC load
    WEATHER_COLLAR = "WEATHER_COLLAR"         # Combined Floor Put + Short Ceiling Call


@dataclass
class WeatherDerivativeContract:
    """Specifications for an exchange-traded or OTC Weather Derivative contract."""
    contract_type: WeatherContractType = WeatherContractType.HDD_SWAP
    base_temperature: float = 18.0             # Standard threshold (18°C / 65°F)
    strike: float = 850.0                      # Strike Degree Days
    floor_strike: Optional[float] = None       # Collar floor strike
    cap_strike: Optional[float] = None         # Collar cap strike
    tick_size_usd: float = 10000.0             # Payment per degree day index unit ($/DD)
    max_payout_cap_usd: float = 2500000.0      # Maximum liability cap ($)
    start_date: str = "2026-11-01"            # Contract start date
    end_date: str = "2027-03-31"              # Contract end date (Winter Season = 151 days)
    discount_rate: float = 0.045               # Risk-free discounting rate (4.5%)


@dataclass
class WeatherPricingResult:
    """Pricing, Greeks, and Risk Analytics for Weather Derivative contracts."""
    contract_type: str
    strike: float
    expected_index_value: float
    index_std_dev: float
    fair_swap_strike: float
    fair_premium_usd: float
    burn_analysis_premium_usd: float
    monte_carlo_premium_usd: float
    expected_payoff_usd: float
    max_payout_cap_usd: float
    var_95_payoff_usd: float
    probability_of_exercise: float

    def summary_table(self) -> pd.DataFrame:
        """Formats weather pricing result into a clean diagnostic table."""
        records = [
            ("Contract Structure", self.contract_type),
            ("Degree Day Strike", f"{self.strike:.1f} DD"),
            ("Expected Index Value E[I]", f"{self.expected_index_value:.1f} DD"),
            ("Index Standard Deviation", f"{self.index_std_dev:.1f} DD"),
            ("Fair Swap Strike", f"{self.fair_swap_strike:.1f} DD"),
            ("Fair Risk-Neutral Premium", f"${self.fair_premium_usd:,.2f}"),
            ("Burn Analysis Premium", f"${self.burn_analysis_premium_usd:,.2f}"),
            ("Monte Carlo Premium (10k paths)", f"${self.monte_carlo_premium_usd:,.2f}"),
            ("Expected Undiscounted Payoff", f"${self.expected_payoff_usd:,.2f}"),
            ("Max Liability Cap", f"${self.max_payout_cap_usd:,.0f}"),
            ("95% Value at Risk (Underwriter)", f"${self.var_95_payoff_usd:,.2f}"),
            ("Probability of Exercise", f"{self.probability_of_exercise:.1%}"),
        ]
        return pd.DataFrame(records, columns=["Metric", "Value"])


class WeatherDerivativePricer:
    """Institutional Pricing Engine for HDD/CDD Weather Swaps and Options."""

    def __init__(self, params: Optional[TemperatureModelParams] = None):
        self.params = params or TemperatureModelParams()

    def deterministic_mean_temp(self, t_day_of_year: np.ndarray, year_fraction: float = 0.0) -> np.ndarray:
        """Calculates deterministic seasonal temperature mean S(t)."""
        A = self.params.base_temp_A
        B = self.params.warming_trend_B
        C = self.params.seasonal_amp_C
        phi = self.params.phase_phi
        return A + B * year_fraction + C * np.sin(2.0 * np.pi * (t_day_of_year - phi) / 365.0)

    def seasonal_volatility(self, t_day_of_year: np.ndarray) -> np.ndarray:
        """Calculates seasonal volatility sigma(t)."""
        s0 = self.params.sigma0
        s1 = self.params.sigma1
        phi_s = self.params.sigma_phase
        return s0 + s1 * np.cos(2.0 * np.pi * (t_day_of_year - phi_s) / 365.0)

    def simulate_temperature_paths(
        self,
        start_date: str,
        end_date: str,
        n_paths: int = 10000,
        use_risk_neutral_mpr: bool = True,
        seed: int = 42,
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """Simulates daily temperature trajectories using calibrated Ornstein-Uhlenbeck SDE."""
        np.random.seed(seed)
        date_range = pd.date_range(start_date, end_date, freq="D")
        n_days = len(date_range)
        day_of_year = date_range.dayofyear.values

        s_det = self.deterministic_mean_temp(day_of_year)
        sig = self.seasonal_volatility(day_of_year)
        kappa = self.params.kappa
        lam = self.params.market_price_of_risk_lambda if use_risk_neutral_mpr else 0.0

        x_paths = np.zeros((n_paths, n_days))
        x_paths[:, 0] = np.random.normal(0, sig[0], n_paths)

        dt = 1.0  # 1 day step
        for t in range(1, n_days):
            drift = -kappa * x_paths[:, t-1] * dt - lam * sig[t-1] * dt
            diffusion = sig[t-1] * np.sqrt(dt) * np.random.normal(0, 1, n_paths)
            x_paths[:, t] = x_paths[:, t-1] + drift + diffusion

        temp_paths = s_det[np.newaxis, :] + x_paths
        return date_range, temp_paths

    def compute_degree_days(
        self,
        temp_series: np.ndarray,
        base_temp: float = 18.0,
        index_type: str = "HDD",
    ) -> np.ndarray:
        """Calculates cumulative HDD or CDD degree days over a temperature path."""
        if index_type.upper() in ["HDD", "HEATING"]:
            daily_dd = np.maximum(0.0, base_temp - temp_series)
        else:  # CDD / COOLING
            daily_dd = np.maximum(0.0, temp_series - base_temp)

        if daily_dd.ndim == 1:
            return np.sum(daily_dd)
        else:
            return np.sum(daily_dd, axis=1)

    def historical_burn_analysis(
        self,
        contract: WeatherDerivativeContract,
        n_historical_years: int = 30,
        seed: int = 42,
    ) -> float:
        """Prices contract via Historical Burn Analysis (HBA) across historical seasons."""
        np.random.seed(seed)
        start_dt = pd.to_datetime(contract.start_date)
        end_dt = pd.to_datetime(contract.end_date)
        duration_days = (end_dt - start_dt).days + 1

        is_hdd = "HDD" in contract.contract_type.value or contract.contract_type == WeatherContractType.WEATHER_COLLAR
        idx_type = "HDD" if is_hdd else "CDD"

        historical_indices = np.zeros(n_historical_years)
        for y in range(n_historical_years):
            s_yr = start_dt.year - n_historical_years + y
            e_yr = s_yr + (1 if end_dt.month < start_dt.month else 0)
            s_str = f"{s_yr}-{start_dt.month:02d}-{start_dt.day:02d}"
            e_str = f"{e_yr}-{end_dt.month:02d}-{end_dt.day:02d}"
            _, temps = self.simulate_temperature_paths(s_str, e_str, n_paths=1, use_risk_neutral_mpr=False, seed=seed + y)
            historical_indices[y] = self.compute_degree_days(temps[0], contract.base_temperature, idx_type)

        payoffs = np.zeros(n_historical_years)
        c_type = contract.contract_type
        K = contract.strike
        tick = contract.tick_size_usd
        cap = contract.max_payout_cap_usd

        for i, idx_val in enumerate(historical_indices):
            if c_type in [WeatherContractType.HDD_CALL, WeatherContractType.CDD_CALL]:
                p = min(cap, tick * max(0.0, idx_val - K))
            elif c_type in [WeatherContractType.HDD_PUT, WeatherContractType.CDD_PUT]:
                p = min(cap, tick * max(0.0, K - idx_val))
            elif c_type == WeatherContractType.WEATHER_COLLAR:
                floor_k = contract.floor_strike or (K - 50.0)
                cap_k = contract.cap_strike or (K + 50.0)
                p_put = min(cap, tick * max(0.0, floor_k - idx_val))
                p_call = min(cap, tick * max(0.0, idx_val - cap_k))
                p = p_put - p_call
            else:  # SWAP
                p = tick * (idx_val - K)
            payoffs[i] = p

        discount = np.exp(-contract.discount_rate * (duration_days / 365.0))
        return float(discount * np.mean(payoffs))

    def price_contract(
        self,
        contract: WeatherDerivativeContract,
        n_mc_sims: int = 10000,
        seed: int = 42,
    ) -> WeatherPricingResult:
        """Prices Weather Derivative contract via Monte Carlo simulation and Burn Analysis."""
        start_dt = pd.to_datetime(contract.start_date)
        end_dt = pd.to_datetime(contract.end_date)
        duration_days = (end_dt - start_dt).days + 1
        t_years = duration_days / 365.0
        discount = np.exp(-contract.discount_rate * t_years)

        is_hdd = "HDD" in contract.contract_type.value or contract.contract_type == WeatherContractType.WEATHER_COLLAR
        idx_type = "HDD" if is_hdd else "CDD"

        # 1. Simulate risk-neutral paths
        _, rn_temps = self.simulate_temperature_paths(
            contract.start_date, contract.end_date, n_paths=n_mc_sims, use_risk_neutral_mpr=True, seed=seed
        )
        rn_indices = self.compute_degree_days(rn_temps, contract.base_temperature, idx_type)

        # 2. Simulate physical paths
        _, phys_temps = self.simulate_temperature_paths(
            contract.start_date, contract.end_date, n_paths=n_mc_sims, use_risk_neutral_mpr=False, seed=seed + 100
        )
        phys_indices = self.compute_degree_days(phys_temps, contract.base_temperature, idx_type)

        exp_index = float(np.mean(phys_indices))
        std_index = float(np.std(phys_indices))
        fair_swap_k = float(np.mean(rn_indices))

        K = contract.strike
        tick = contract.tick_size_usd
        cap = contract.max_payout_cap_usd
        c_type = contract.contract_type

        # Compute payoffs on risk-neutral paths
        if c_type in [WeatherContractType.HDD_CALL, WeatherContractType.CDD_CALL]:
            rn_payoffs = np.minimum(cap, tick * np.maximum(0.0, rn_indices - K))
            phys_payoffs = np.minimum(cap, tick * np.maximum(0.0, phys_indices - K))
            p_exercise = float(np.mean(phys_indices > K))

        elif c_type in [WeatherContractType.HDD_PUT, WeatherContractType.CDD_PUT]:
            rn_payoffs = np.minimum(cap, tick * np.maximum(0.0, K - rn_indices))
            phys_payoffs = np.minimum(cap, tick * np.maximum(0.0, K - phys_indices))
            p_exercise = float(np.mean(phys_indices < K))

        elif c_type == WeatherContractType.WEATHER_COLLAR:
            floor_k = contract.floor_strike or (K - 50.0)
            cap_k = contract.cap_strike or (K + 50.0)
            p_put_rn = np.minimum(cap, tick * np.maximum(0.0, floor_k - rn_indices))
            p_call_rn = np.minimum(cap, tick * np.maximum(0.0, rn_indices - cap_k))
            rn_payoffs = p_put_rn - p_call_rn

            p_put_ph = np.minimum(cap, tick * np.maximum(0.0, floor_k - phys_indices))
            p_call_ph = np.minimum(cap, tick * np.maximum(0.0, phys_indices - cap_k))
            phys_payoffs = p_put_ph - p_call_ph
            p_exercise = float(np.mean((phys_indices < floor_k) | (phys_indices > cap_k)))

        else:  # SWAP
            rn_payoffs = tick * (rn_indices - K)
            phys_payoffs = tick * (phys_indices - K)
            p_exercise = 1.0

        mc_premium = float(discount * np.mean(rn_payoffs))
        exp_payoff = float(np.mean(phys_payoffs))
        burn_premium = self.historical_burn_analysis(contract, n_historical_years=25, seed=seed)
        var_95 = float(np.percentile(phys_payoffs, 95.0))

        return WeatherPricingResult(
            contract_type=c_type.value,
            strike=K,
            expected_index_value=exp_index,
            index_std_dev=std_index,
            fair_swap_strike=fair_swap_k,
            fair_premium_usd=mc_premium,
            burn_analysis_premium_usd=burn_premium,
            monte_carlo_premium_usd=mc_premium,
            expected_payoff_usd=exp_payoff,
            max_payout_cap_usd=cap,
            var_95_payoff_usd=var_95,
            probability_of_exercise=p_exercise,
        )
