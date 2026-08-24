"""Binomial Tree (Lattice) option pricer for European and American options."""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class LatticeGreeks:
    delta: float
    gamma: float
    theta: float  # annualized
    theta_daily: float  # per calendar day (/365)


@dataclass
class TreeNode:
    step: int
    up_steps: int
    stock_price: float
    option_value: float
    continuation_value: float
    intrinsic_value: float
    early_exercised: bool


@dataclass
class BinomialPriceResult:
    price: float
    option_type: str
    exercise_style: str
    model: str
    n_steps: int
    european_price: float
    american_price: float
    early_exercise_premium: float
    greeks: LatticeGreeks
    u: float
    d: float
    p: float
    dt: float
    discount_factor: float


class BinomialTreePricer:
    """Binomial Option Pricing Model supporting CRR, Jarrow-Rudd, and Leisen-Reimer lattices.

    Supports both European and American exercise styles with exact lattice Greek extraction.
    """

    def __init__(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float = 0.0,
        n_steps: int = 100,
    ) -> None:
        if S0 <= 0.0:
            raise ValueError(f"S0 must be positive, got {S0}")
        if K <= 0.0:
            raise ValueError(f"K must be positive, got {K}")
        if T <= 0.0:
            raise ValueError(f"T must be positive, got {T}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.q = float(q)
        self.n_steps = int(n_steps)

    def _peizer_pratt(self, z: float, n: int) -> float:
        """Peizer-Pratt inversion formula for Leisen-Reimer tree."""
        term = (z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))) ** 2 * (n + 1.0 / 6.0)
        h = 0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-term))
        return float(np.clip(h, 1e-8, 1.0 - 1e-8))

    def get_lattice_parameters(self, model: str = "crr", n_steps: Optional[int] = None) -> Tuple[float, float, float, float]:
        """Computes up-factor (u), down-factor (d), risk-neutral probability (p), and dt."""
        N = n_steps if n_steps is not None else self.n_steps
        dt = self.T / N
        sigma = self.sigma
        r = self.r
        q = self.q

        mod = model.lower()
        if mod == "crr":
            # Cox-Ross-Rubinstein
            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u
            p = (np.exp((r - q) * dt) - d) / (u - d)
        elif mod == "jr":
            # Jarrow-Rudd (Equal Probability)
            nu = r - q - 0.5 * sigma ** 2
            u = np.exp(nu * dt + sigma * np.sqrt(dt))
            d = np.exp(nu * dt - sigma * np.sqrt(dt))
            p = 0.5
        elif mod == "lr":
            # Leisen-Reimer (Accelerated Convergence)
            # Ensure odd number of steps if possible for best centering
            d1 = (np.log(self.S0 / self.K) + (r - q + 0.5 * sigma ** 2) * self.T) / (sigma * np.sqrt(self.T))
            d2 = d1 - sigma * np.sqrt(self.T)
            p = self._peizer_pratt(d2, N)
            p_prime = self._peizer_pratt(d1, N)
            u = np.exp((r - q) * dt) * (p_prime / p)
            d = (np.exp((r - q) * dt) - p * u) / (1.0 - p)
        else:
            raise ValueError(f"Unknown binomial model: {model}. Supported: 'crr', 'jr', 'lr'.")

        p = float(np.clip(p, 0.0, 1.0))
        return float(u), float(d), float(p), float(dt)

    def price(
        self,
        option_type: str = "call",
        exercise_style: str = "european",
        model: str = "crr",
        n_steps: Optional[int] = None,
    ) -> BinomialPriceResult:
        """Prices European or American option using backward induction on binomial lattice."""
        N = n_steps if n_steps is not None else self.n_steps
        if N < 2:
            N = 2  # require at least 2 steps for Greeks extraction

        u, d, p, dt = self.get_lattice_parameters(model=model, n_steps=N)
        disc = np.exp(-self.r * dt)
        is_call = option_type.lower() == "call"
        is_american = exercise_style.lower() == "american"

        # Terminal stock prices at step N: j up-moves (0 <= j <= N)
        j_indices = np.arange(N + 1)
        S_terminal = self.S0 * (u ** j_indices) * (d ** (N - j_indices))

        # Terminal payoffs
        if is_call:
            V_eur = np.maximum(S_terminal - self.K, 0.0)
            V_ame = np.maximum(S_terminal - self.K, 0.0)
        else:
            V_eur = np.maximum(self.K - S_terminal, 0.0)
            V_ame = np.maximum(self.K - S_terminal, 0.0)

        # Storage for Greeks extraction at step 1 and step 2
        V_step2 = None
        S_step2 = None
        V_step1 = None
        S_step1 = None

        # Backward Induction
        for i in range(N - 1, -1, -1):
            j_curr = np.arange(i + 1)
            S_curr = self.S0 * (u ** j_curr) * (d ** (i - j_curr))

            # European continuation
            V_eur = disc * (p * V_eur[1:] + (1.0 - p) * V_eur[:-1])

            # American continuation vs immediate exercise
            continuation_ame = disc * (p * V_ame[1:] + (1.0 - p) * V_ame[:-1])
            if is_call:
                intrinsic = np.maximum(S_curr - self.K, 0.0)
            else:
                intrinsic = np.maximum(self.K - S_curr, 0.0)

            V_ame = np.maximum(intrinsic, continuation_ame)

            if i == 2:
                V_step2 = V_ame.copy() if is_american else V_eur.copy()
                S_step2 = S_curr.copy()
            elif i == 1:
                V_step1 = V_ame.copy() if is_american else V_eur.copy()
                S_step1 = S_curr.copy()

        price_eur = float(V_eur[0])
        price_ame = float(V_ame[0])
        target_price = price_ame if is_american else price_eur
        early_premium = float(max(price_ame - price_eur, 0.0))

        # Greeks extraction from node values
        # Delta at t=0 (between step 1 nodes)
        delta = float((V_step1[1] - V_step1[0]) / (S_step1[1] - S_step1[0]))

        # Gamma at t=0 (using step 2 nodes)
        delta_up = (V_step2[2] - V_step2[1]) / (S_step2[2] - S_step2[1])
        delta_down = (V_step2[1] - V_step2[0]) / (S_step2[1] - S_step2[0])
        gamma = float((delta_up - delta_down) / (0.5 * (S_step2[2] - S_step2[0])))

        # Theta at t=0 (difference between step 2 center node and step 0)
        theta_ann = float((V_step2[1] - target_price) / (2.0 * dt))
        theta_daily = float(theta_ann / 365.0)

        greeks = LatticeGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta_ann,
            theta_daily=theta_daily,
        )

        return BinomialPriceResult(
            price=target_price,
            option_type=option_type.lower(),
            exercise_style=exercise_style.lower(),
            model=model.lower(),
            n_steps=N,
            european_price=price_eur,
            american_price=price_ame,
            early_exercise_premium=early_premium,
            greeks=greeks,
            u=u,
            d=d,
            p=p,
            dt=dt,
            discount_factor=disc,
        )

    def build_tree(
        self,
        n_steps: int = 3,
        option_type: str = "call",
        exercise_style: str = "american",
        model: str = "crr",
    ) -> List[List[TreeNode]]:
        """Constructs explicit tree representation for inspection or visualization."""
        u, d, p, dt = self.get_lattice_parameters(model=model, n_steps=n_steps)
        disc = np.exp(-self.r * dt)
        is_call = option_type.lower() == "call"
        is_american = exercise_style.lower() == "american"

        # Build stock price lattice
        stock_tree = []
        for i in range(n_steps + 1):
            row = [self.S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)]
            stock_tree.append(row)

        # Terminal payoffs
        opt_tree = [[] for _ in range(n_steps + 1)]
        for j in range(n_steps + 1):
            S_val = stock_tree[n_steps][j]
            intr = max(S_val - self.K, 0.0) if is_call else max(self.K - S_val, 0.0)
            opt_tree[n_steps].append(intr)

        # Backward roll
        cont_tree = [[] for _ in range(n_steps + 1)]
        exercised_tree = [[] for _ in range(n_steps + 1)]
        cont_tree[n_steps] = opt_tree[n_steps].copy()
        exercised_tree[n_steps] = [False] * (n_steps + 1)

        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                S_val = stock_tree[i][j]
                cont_val = disc * (p * opt_tree[i + 1][j + 1] + (1.0 - p) * opt_tree[i + 1][j])
                intr = max(S_val - self.K, 0.0) if is_call else max(self.K - S_val, 0.0)

                if is_american:
                    val = max(intr, cont_val)
                    early_ex = (intr > cont_val) and (intr > 0.0)
                else:
                    val = cont_val
                    early_ex = False

                opt_tree[i].append(val)
                cont_tree[i].append(cont_val)
                exercised_tree[i].append(early_ex)

        # Assemble TreeNode objects
        full_tree = []
        for i in range(n_steps + 1):
            row_nodes = []
            for j in range(i + 1):
                S_val = stock_tree[i][j]
                opt_val = opt_tree[i][j]
                cont_val = cont_tree[i][j]
                intr = max(S_val - self.K, 0.0) if is_call else max(self.K - S_val, 0.0)
                early_ex = exercised_tree[i][j]
                node = TreeNode(
                    step=i,
                    up_steps=j,
                    stock_price=float(S_val),
                    option_value=float(opt_val),
                    continuation_value=float(cont_val),
                    intrinsic_value=float(intr),
                    early_exercised=bool(early_ex),
                )
                row_nodes.append(node)
            full_tree.append(row_nodes)

        return full_tree
