"""Walk-forward validation and overfitting diagnostics for systematic strategies.

Implements:
- In-Sample (IS) vs Out-of-Sample (OOS) Train/Test Splitting
- Rolling & Expanding Walk-Forward Backtesting
- Performance Degradation Metrics (IS Sharpe vs OOS Sharpe)
- Overfitting Detection & Parameter Sensitivity Audits
"""

from typing import Callable, Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .backtester import BacktestEngine, BacktestResult
from .costs import TransactionCostModel


@dataclass
class WalkForwardFoldResult:
    """Results for a single walk-forward train/test slice."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    in_sample_metrics: Dict[str, Any]
    out_of_sample_metrics: Dict[str, Any]
    oos_returns: pd.Series
    is_sharpe: float
    oos_sharpe: float
    sharpe_degradation_pct: float


@dataclass
class WalkForwardReport:
    """Aggregated walk-forward validation diagnostics."""
    folds: List[WalkForwardFoldResult]
    stitched_oos_returns: pd.Series
    stitched_oos_metrics: Dict[str, Any]
    mean_is_sharpe: float
    mean_oos_sharpe: float
    overall_sharpe_degradation: float
    overfitting_risk_verdict: str

    def summary_table(self) -> pd.DataFrame:
        """Generates a fold-by-fold comparison table."""
        records = []
        for f in self.folds:
            records.append({
                "Fold": f.fold_index,
                "Train Window": f"{f.train_start.strftime('%Y-%m')} to {f.train_end.strftime('%Y-%m')}",
                "Test Window": f"{f.test_start.strftime('%Y-%m')} to {f.test_end.strftime('%Y-%m')}",
                "IS Sharpe": f"{f.is_sharpe:.2f}",
                "OOS Sharpe": f"{f.oos_sharpe:.2f}",
                "Degradation": f"{f.sharpe_degradation_pct:.1%}",
                "OOS CAGR": f"{f.out_of_sample_metrics.get('cagr', 0.0):+.2%}",
                "OOS MaxDD": f"{f.out_of_sample_metrics.get('max_drawdown', 0.0):.2%}",
            })
        return pd.DataFrame(records)

    def print_report(self) -> None:
        """Prints a clean terminal validation card."""
        print("┌" + "─" * 78 + "┐")
        print("│" + "WALK-FORWARD ROBUSTNESS & OVERFITTING AUDIT".center(78) + "│")
        print("├" + "─" * 78 + "┤")
        print(f"│  Number of Folds Evaluated:     {len(self.folds):<43} │")
        print(f"│  Average In-Sample Sharpe:      {self.mean_is_sharpe:<43.2f} │")
        print(f"│  Average Out-of-Sample Sharpe:  {self.mean_oos_sharpe:<43.2f} │")
        print(f"│  Overall Sharpe Degradation:    {self.overall_sharpe_degradation:<43.1%} │")
        print(f"│  Stitched OOS Total Return:     {self.stitched_oos_metrics.get('total_return', 0.0):<+43.2%} │")
        print(f"│  Stitched OOS Annual Volatility:{self.stitched_oos_metrics.get('annualized_volatility', 0.0):<43.2%} │")
        print(f"│  Stitched OOS Max Drawdown:     {self.stitched_oos_metrics.get('max_drawdown', 0.0):<43.2%} │")
        print(f"│  OVERFITTING RISK VERDICT:      {self.overfitting_risk_verdict:<43} │")
        print("├" + "─" * 78 + "┤")
        print("│ FOLD SUMMARY:" + " " * 65 + "│")
        for f in self.folds:
            line = f"Fold {f.fold_index} | IS Sharpe: {f.is_sharpe:>4.2f} -> OOS Sharpe: {f.oos_sharpe:>4.2f} (Degradation: {f.sharpe_degradation_pct:>5.1%})"
            print(f"│   {line:<74} │")
        print("└" + "─" * 78 + "┘")


class WalkForwardValidator:
    """Walk-forward train/test validator and curve-fitting detector."""

    def __init__(
        self,
        train_window_days: int = 504,  # 2 years
        test_window_days: int = 126,   # 6 months
        step_days: Optional[int] = None,
        expanding_window: bool = False,
        cost_model: Optional[TransactionCostModel] = None,
        risk_free_rate: float = 0.02,
    ) -> None:
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days if step_days is not None else test_window_days
        self.expanding_window = expanding_window
        self.backtester = BacktestEngine(cost_model=cost_model, risk_free_rate=risk_free_rate)

    def simple_train_test_split(
        self,
        prices_or_returns: Union[pd.Series, pd.DataFrame],
        weights: Union[pd.Series, pd.DataFrame],
        train_ratio: float = 0.70,
    ) -> Tuple[BacktestResult, BacktestResult, float]:
        """Performs a single chronological In-Sample vs Out-of-Sample train/test split.

        Returns:
            is_result: BacktestResult on training data.
            oos_result: BacktestResult on held-out test data.
            sharpe_degradation: Relative drop from IS Sharpe to OOS Sharpe.
        """
        common_idx = prices_or_returns.index.intersection(weights.index)
        split_point = int(len(common_idx) * train_ratio)

        is_idx = common_idx[:split_point]
        oos_idx = common_idx[split_point:]

        is_res = self.backtester.run(
            prices_or_returns.loc[is_idx],
            weights.loc[is_idx],
            strategy_name="In-Sample (Train)",
        )
        oos_res = self.backtester.run(
            prices_or_returns.loc[oos_idx],
            weights.loc[oos_idx],
            strategy_name="Out-of-Sample (Test)",
        )

        is_sharpe = is_res.metrics.get("sharpe_ratio", 0.0)
        oos_sharpe = oos_res.metrics.get("sharpe_ratio", 0.0)

        if is_sharpe > 0:
            degradation = max(0.0, 1.0 - (oos_sharpe / is_sharpe))
        else:
            degradation = 1.0 if oos_sharpe < is_sharpe else 0.0

        return is_res, oos_res, float(degradation)

    def walk_forward_evaluate(
        self,
        strategy_fit_func: Callable[[Union[pd.Series, pd.DataFrame]], Union[pd.Series, pd.DataFrame]],
        prices: Union[pd.Series, pd.DataFrame],
    ) -> WalkForwardReport:
        """Runs rolling walk-forward backtesting.

        Args:
            strategy_fit_func: Function taking training price data and returning strategy weights for any input data.
            prices: Historical price DataFrame or Series.

        Returns:
            WalkForwardReport containing diagnostics and stitched OOS performance.
        """
        n_days = len(prices)
        dates = prices.index
        fold_results = []
        all_oos_returns = []

        start_idx = 0
        fold_idx = 1

        while start_idx + self.train_window_days + self.test_window_days <= n_days:
            train_start_i = 0 if self.expanding_window else start_idx
            train_end_i = start_idx + self.train_window_days
            test_end_i = train_end_i + self.test_window_days

            train_prices = prices.iloc[train_start_i:train_end_i]
            test_prices = prices.iloc[train_end_i:test_end_i]

            # Fit strategy on In-Sample data
            is_weights = strategy_fit_func(train_prices)
            is_res = self.backtester.run(train_prices, is_weights, strategy_name=f"Fold {fold_idx} IS")

            # Evaluate on Out-of-Sample data
            oos_weights = strategy_fit_func(test_prices)
            oos_res = self.backtester.run(test_prices, oos_weights, strategy_name=f"Fold {fold_idx} OOS")

            is_sharpe = is_res.metrics.get("sharpe_ratio", 0.0)
            oos_sharpe = oos_res.metrics.get("sharpe_ratio", 0.0)

            if is_sharpe > 0:
                deg = max(0.0, 1.0 - (oos_sharpe / is_sharpe))
            else:
                deg = 1.0 if oos_sharpe < is_sharpe else 0.0

            fold_results.append(WalkForwardFoldResult(
                fold_index=fold_idx,
                train_start=dates[train_start_i],
                train_end=dates[train_end_i - 1],
                test_start=dates[train_end_i],
                test_end=dates[test_end_i - 1],
                in_sample_metrics=is_res.metrics,
                out_of_sample_metrics=oos_res.metrics,
                oos_returns=oos_res.net_returns,
                is_sharpe=float(is_sharpe),
                oos_sharpe=float(oos_sharpe),
                sharpe_degradation_pct=float(deg),
            ))

            all_oos_returns.append(oos_res.net_returns)
            start_idx += self.step_days
            fold_idx += 1

        if len(all_oos_returns) > 0:
            stitched_returns = pd.concat(all_oos_returns)
            stitched_res = self.backtester.run(
                stitched_returns,
                pd.Series(1.0, index=stitched_returns.index),
                is_returns=True,
                strategy_name="Stitched OOS Walk-Forward",
            )
            stitched_metrics = stitched_res.metrics
        else:
            stitched_returns = pd.Series(dtype=float)
            stitched_metrics = {}

        mean_is = float(np.mean([f.is_sharpe for f in fold_results])) if fold_results else 0.0
        mean_oos = float(np.mean([f.oos_sharpe for f in fold_results])) if fold_results else 0.0

        if mean_is > 0:
            overall_deg = float(max(0.0, 1.0 - (mean_oos / mean_is)))
        else:
            overall_deg = 0.0

        # Overfitting verdict
        if mean_oos >= 0.8 * mean_is and mean_oos > 0.6:
            verdict = "PASSED (Robust Strategy - Low Overfitting Risk)"
        elif mean_oos >= 0.5 * mean_is and mean_oos > 0.3:
            verdict = "MODERATE (Noticeable Degradation - Monitor Closely)"
        else:
            verdict = "HIGH RISK (Severe In-Sample Overfitting Detected)"

        return WalkForwardReport(
            folds=fold_results,
            stitched_oos_returns=stitched_returns,
            stitched_oos_metrics=stitched_metrics,
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            overall_sharpe_degradation=overall_deg,
            overfitting_risk_verdict=verdict,
        )
