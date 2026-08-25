"""Carbon Allowance Pricing & ETS Fuel-Switching Dynamics (Project 36).

Implements the fundamental economic framework of the European Union Emissions Trading System (EU ETS):
1. Clean Spark Spread (CSS) and Clean Dark Spread (CDS) power generation margin calculations.
2. Theoretical Fuel-Switching Carbon Price Parity ($P_{\text{switch}}$).
3. Mean-reverting jump-diffusion simulation for carbon allowance prices (EUA) with policy jump shocks.
4. Carbon futures curve modeling with convenience yield and cost-of-carry.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class CleanSpreadsResult:
    """Dataclass holding Clean Spark and Clean Dark Spread values."""
    power_price: float
    gas_price: float
    coal_price: float
    carbon_price: float
    clean_spark_spread: float
    clean_dark_spread: float
    spark_dark_differential: float
    fuel_switch_parity_price: float
    dominant_fuel: str
    carbon_sensitivity_gas: float
    carbon_sensitivity_coal: float


@dataclass
class FuelSwitchingParity:
    """Dataclass holding theoretical fuel-switching equilibrium analytics."""
    gas_price_thermal: float
    coal_price_thermal: float
    efficiency_gas: float
    efficiency_coal: float
    emission_factor_gas: float
    emission_factor_coal: float
    switching_carbon_price: float
    efficiency_ratio: float
    emission_spread: float

    def summary(self) -> Dict[str, Any]:
        return {
            "Gas Price (€/MWh_th)": self.gas_price_thermal,
            "Coal Price (€/MWh_th)": self.coal_price_thermal,
            "Gas Efficiency (η_gas)": f"{self.efficiency_gas:.1%}",
            "Coal Efficiency (η_coal)": f"{self.efficiency_coal:.1%}",
            "Gas Emission Factor (tCO2/MWh_e)": self.emission_factor_gas,
            "Coal Emission Factor (tCO2/MWh_e)": self.emission_factor_coal,
            "Theoretical Fuel-Switching Parity (€/tCO2)": round(self.switching_carbon_price, 2),
            "Emission Differential (tCO2/MWh_e)": round(self.emission_spread, 3),
        }


@dataclass
class CarbonSimulationResult:
    """Dataclass containing simulated EUA allowance paths and statistics."""
    time_grid: np.ndarray
    paths: np.ndarray
    expected_path: np.ndarray
    upper_95_path: np.ndarray
    lower_95_path: np.ndarray
    terminal_prices: np.ndarray
    mean_terminal_price: float
    median_terminal_price: float
    terminal_volatility: float
    jump_occurrences: int


class CarbonAllowanceModel:
    """Comprehensive EU ETS Carbon Allowance & Fuel-Switching Valuation Engine.
    
    Attributes:
        default_eff_gas: CCGT Gas thermal electrical efficiency (default 50%).
        default_eff_coal: Coal thermal electrical efficiency (default 38%).
        default_ef_gas: Gas emission factor per MWh electrical (default 0.37 tCO2/MWh_e).
        default_ef_coal: Coal emission factor per MWh electrical (default 0.95 tCO2/MWh_e).
    """

    def __init__(
        self,
        efficiency_gas: float = 0.50,
        efficiency_coal: float = 0.38,
        emission_factor_gas: float = 0.37,
        emission_factor_coal: float = 0.95,
    ):
        self.efficiency_gas = efficiency_gas
        self.efficiency_coal = efficiency_coal
        self.emission_factor_gas = emission_factor_gas
        self.emission_factor_coal = emission_factor_coal

    # =========================================================================
    # 1. CLEAN SPREAD CALCULATIONS
    # =========================================================================

    def compute_clean_spark_spread(
        self,
        power_price: Union[float, np.ndarray, pd.Series],
        gas_price: Union[float, np.ndarray, pd.Series],
        carbon_price: Union[float, np.ndarray, pd.Series],
        efficiency_gas: Optional[float] = None,
        emission_factor_gas: Optional[float] = None,
    ) -> Union[float, np.ndarray, pd.Series]:
        """Calculates the Clean Spark Spread (CSS) in €/MWh_e for natural gas generation.
        
        CSS = Power_Price - (Gas_Price / η_gas) - (EF_gas * Carbon_Price)
        """
        eta_g = efficiency_gas if efficiency_gas is not None else self.efficiency_gas
        ef_g = emission_factor_gas if emission_factor_gas is not None else self.emission_factor_gas
        
        fuel_cost_per_mwhe = gas_price / eta_g
        carbon_cost_per_mwhe = ef_g * carbon_price
        css = power_price - fuel_cost_per_mwhe - carbon_cost_per_mwhe
        return css

    def compute_clean_dark_spread(
        self,
        power_price: Union[float, np.ndarray, pd.Series],
        coal_price: Union[float, np.ndarray, pd.Series],
        carbon_price: Union[float, np.ndarray, pd.Series],
        efficiency_coal: Optional[float] = None,
        emission_factor_coal: Optional[float] = None,
    ) -> Union[float, np.ndarray, pd.Series]:
        """Calculates the Clean Dark Spread (CDS) in €/MWh_e for coal generation.
        
        CDS = Power_Price - (Coal_Price / η_coal) - (EF_coal * Carbon_Price)
        """
        eta_c = efficiency_coal if efficiency_coal is not None else self.efficiency_coal
        ef_c = emission_factor_coal if emission_factor_coal is not None else self.emission_factor_coal
        
        fuel_cost_per_mwhe = coal_price / eta_c
        carbon_cost_per_mwhe = ef_c * carbon_price
        cds = power_price - fuel_cost_per_mwhe - carbon_cost_per_mwhe
        return cds

    def compute_fuel_switching_price(
        self,
        gas_price: Union[float, np.ndarray, pd.Series],
        coal_price: Union[float, np.ndarray, pd.Series],
        efficiency_gas: Optional[float] = None,
        efficiency_coal: Optional[float] = None,
        emission_factor_gas: Optional[float] = None,
        emission_factor_coal: Optional[float] = None,
    ) -> Union[float, np.ndarray, pd.Series]:
        """Calculates the theoretical fuel-switching carbon price parity ($P_{\text{switch}}$).
        
        At parity, CSS = CDS:
        P_switch = [(P_gas / η_gas) - (P_coal / η_coal)] / (EF_coal - EF_gas)
        """
        eta_g = efficiency_gas if efficiency_gas is not None else self.efficiency_gas
        eta_c = efficiency_coal if efficiency_coal is not None else self.efficiency_coal
        ef_g = emission_factor_gas if emission_factor_gas is not None else self.emission_factor_gas
        ef_c = emission_factor_coal if emission_factor_coal is not None else self.emission_factor_coal

        emission_diff = ef_c - ef_g
        if isinstance(emission_diff, (int, float)) and emission_diff <= 0:
            raise ValueError("Coal emission factor must strictly exceed gas emission factor.")

        fuel_diff = (gas_price / eta_g) - (coal_price / eta_c)
        switching_price = fuel_diff / emission_diff
        return switching_price

    def evaluate_spread_snapshot(
        self,
        power_price: float,
        gas_price: float,
        coal_price: float,
        carbon_price: float,
    ) -> CleanSpreadsResult:
        """Evaluates single-point economic spreads and fuel merit order."""
        css = float(self.compute_clean_spark_spread(power_price, gas_price, carbon_price))
        cds = float(self.compute_clean_dark_spread(power_price, coal_price, carbon_price))
        p_switch = float(self.compute_fuel_switching_price(gas_price, coal_price))
        diff = css - cds
        dominant = "Gas (CCGT)" if diff > 0 else "Coal (Thermal)"

        return CleanSpreadsResult(
            power_price=power_price,
            gas_price=gas_price,
            coal_price=coal_price,
            carbon_price=carbon_price,
            clean_spark_spread=round(css, 2),
            clean_dark_spread=round(cds, 2),
            spark_dark_differential=round(diff, 2),
            fuel_switch_parity_price=round(p_switch, 2),
            dominant_fuel=dominant,
            carbon_sensitivity_gas=-self.emission_factor_gas,
            carbon_sensitivity_coal=-self.emission_factor_coal,
        )

    def compute_spreads_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes CSS, CDS, and Fuel-Switching Parity series across a market dataset."""
        res_df = df.copy()
        res_df["CSS"] = self.compute_clean_spark_spread(
            power_price=res_df["Power_Baseload"],
            gas_price=res_df["Gas_TTF"],
            carbon_price=res_df["EUA_Carbon_Spot"]
        )
        res_df["CDS"] = self.compute_clean_dark_spread(
            power_price=res_df["Power_Baseload"],
            coal_price=res_df["Coal_ARA"],
            carbon_price=res_df["EUA_Carbon_Spot"]
        )
        res_df["P_Switch_Parity"] = self.compute_fuel_switching_price(
            gas_price=res_df["Gas_TTF"],
            coal_price=res_df["Coal_ARA"]
        )
        res_df["Spark_Dark_Diff"] = res_df["CSS"] - res_df["CDS"]
        res_df["Switching_Incentive"] = np.where(res_df["EUA_Carbon_Spot"] > res_df["P_Switch_Parity"], "Gas Favorable", "Coal Favorable")
        return res_df

    # =========================================================================
    # 2. MEAN-REVERTING JUMP-DIFFUSION STOCHASTIC SIMULATION
    # =========================================================================

    def simulate_jump_diffusion(
        self,
        s0: float = 80.0,
        horizon_years: float = 1.0,
        n_steps: int = 252,
        n_paths: int = 1000,
        kappa: float = 1.2,
        theta: float = 85.0,
        sigma: float = 0.35,
        jump_intensity: float = 4.0,  # ~4 policy jumps per year
        mu_jump: float = 0.05,
        sigma_jump: float = 0.12,
        seed: int = 42,
    ) -> CarbonSimulationResult:
        """Simulates EUA allowance price trajectories using a Mean-Reverting Jump Diffusion process.
        
        d ln(S_t) = kappa * (ln(theta) - ln(S_t)) dt + sigma dW_t + J_t dN_t
        """
        rng = np.random.default_rng(seed)
        dt = horizon_years / n_steps
        time_grid = np.linspace(0, horizon_years, n_steps + 1)

        log_paths = np.zeros((n_steps + 1, n_paths))
        log_paths[0, :] = np.log(s0)

        log_theta = np.log(theta)
        total_jumps = 0

        for t in range(1, n_steps + 1):
            drift = kappa * (log_theta - log_paths[t-1, :]) * dt
            diffusion = sigma * np.sqrt(dt) * rng.normal(0, 1.0, n_paths)
            
            # Poisson jump events
            n_jumps = rng.poisson(jump_intensity * dt, n_paths)
            total_jumps += int(np.sum(n_jumps))
            
            jump_sizes = np.zeros(n_paths)
            has_jump = n_jumps > 0
            if np.any(has_jump):
                for idx in np.where(has_jump)[0]:
                    jump_sizes[idx] = np.sum(rng.normal(mu_jump, sigma_jump, n_jumps[idx]))

            log_paths[t, :] = log_paths[t-1, :] + drift + diffusion + jump_sizes

        price_paths = np.exp(log_paths)
        expected_path = np.mean(price_paths, axis=1)
        upper_95 = np.percentile(price_paths, 97.5, axis=1)
        lower_95 = np.percentile(price_paths, 2.5, axis=1)
        terminal = price_paths[-1, :]

        return CarbonSimulationResult(
            time_grid=time_grid,
            paths=price_paths,
            expected_path=expected_path,
            upper_95_path=upper_95,
            lower_95_path=lower_95,
            terminal_prices=terminal,
            mean_terminal_price=float(np.mean(terminal)),
            median_terminal_price=float(np.median(terminal)),
            terminal_volatility=float(np.std(terminal) / np.mean(terminal)),
            jump_occurrences=total_jumps,
        )

    # =========================================================================
    # 3. JUMP DIFFUSION CALIBRATION
    # =========================================================================

    def calibrate_jump_diffusion(
        self,
        prices_series: pd.Series,
        dt: float = 1.0 / 252.0,
    ) -> Dict[str, float]:
        """Calibrates mean-reverting jump-diffusion parameters from historical EUA price series."""
        log_prices = np.log(prices_series.dropna().values)
        d_log = np.diff(log_prices)
        x_lag = log_prices[:-1]

        # Step 1: OLS regression for mean-reversion drift: d_log = alpha + beta * x_lag
        beta, alpha = np.polyfit(x_lag, d_log, 1)
        kappa = -beta / dt
        theta = np.exp(-alpha / beta) if beta != 0 else np.exp(np.mean(log_prices))
        kappa = max(0.01, min(10.0, kappa))

        residuals = d_log - (alpha + beta * x_lag)
        
        # Step 2: Separate normal diffusion from tail jumps via 3-sigma outlier filtering
        std_res = np.std(residuals)
        is_jump = np.abs(residuals) > 2.5 * std_res
        
        diff_residuals = residuals[~is_jump]
        jump_residuals = residuals[is_jump]

        sigma = float(np.std(diff_residuals) / np.sqrt(dt))
        jump_count = len(jump_residuals)
        total_time_years = len(log_prices) * dt
        jump_intensity = float(jump_count / total_time_years)

        mu_jump = float(np.mean(jump_residuals)) if len(jump_residuals) > 0 else 0.0
        sigma_jump = float(np.std(jump_residuals)) if len(jump_residuals) > 0 else 0.05

        return {
            "kappa": round(kappa, 4),
            "theta": round(float(theta), 2),
            "sigma": round(sigma, 4),
            "jump_intensity_annual": round(jump_intensity, 2),
            "mu_jump": round(mu_jump, 4),
            "sigma_jump": round(sigma_jump, 4),
            "half_life_days": round(float(np.log(2.0) / (kappa * dt)), 1),
        }

    # =========================================================================
    # 4. CARBON FUTURES CURVE & COST OF CARRY
    # =========================================================================

    def construct_futures_curve(
        self,
        spot_price: float = 80.0,
        tenors_years: Optional[List[float]] = None,
        r: float = 0.03,
        convenience_yield: float = 0.01,
        storage_cost: float = 0.002,
    ) -> pd.DataFrame:
        """Constructs theoretical EUA carbon futures curve under cost-of-carry framework.
        
        F(0, T) = S_0 * exp((r + u - y) * T)
        """
        if tenors_years is None:
            tenors_years = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]

        tenors = np.array(tenors_years)
        net_carry = r + storage_cost - convenience_yield
        futures_prices = spot_price * np.exp(net_carry * tenors)

        return pd.DataFrame({
            "Tenor_Years": tenors,
            "Spot_Price": spot_price,
            "Futures_Price": np.round(futures_prices, 2),
            "Basis_EUR": np.round(futures_prices - spot_price, 2),
            "Basis_Pct": np.round(((futures_prices / spot_price) - 1.0) * 100.0, 2),
            "Annualized_Roll_Yield_Pct": np.round((convenience_yield - r - storage_cost) * 100.0, 2),
        })
