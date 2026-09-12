\# Nexus Quant: Core Algorithms, Mathematical Formulations & Trading Applications Guide

\#\# An Exhaustive Technical Reference for All 11 Quantitative Tracks (Projects 01–55)

&nbsp;

\---

&nbsp;

\#\# Executive Summary & Structure

This guide details the mathematical foundations, computational workflows, and practical trading applications for all 11 foundational quantitative finance tracks implemented in the \*\*Nexus Quant Platform\*\*.

&nbsp;

For each track, three core dimensions are documented:

1\. \*\*The Core Formulae & Algorithms\*\*: Exact continuous and discrete mathematical formulations.

2\. \*\*How These Are Used\*\*: Data requirements, algorithmic steps, and computational mechanics.

3\. \*\*Trading Applications & Buy/Sell Decisions\*\*: Concrete implementation across Equities, Fixed Income, FX, Commodities, Derivatives, Crypto, and Event/Prediction Markets.

&nbsp;

\---

&nbsp;

\#\# Track 1: Market Data & Quant Foundations (Projects 01–05)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Logarithmic & Compounded Returns\*\*:

  $$r\_t \= \\ln\\left(\\frac{P\_t}{P\_{t-1}}\\right), \\quad R\_t \= \\frac{P\_t \- P\_{t-1}}{P\_{t-1}}, \\quad \\text{CAGR} \= \\exp\\left(\\frac{252}{T}\\sum\_{t=1}^T r\_t\\right) \- 1$$

\- \*\*Range-Based Volatility Estimators\*\*:

  \- \*Parkinson (1980)\* (High-Low Range):

    $$\\sigma\_P \= \\sqrt{\\frac{252}{4 \\ln 2 \\cdot N} \\sum\_{t=1}^N \\left(\\ln \\frac{H\_t}{L\_t}\\right)^2}$$

  \- \*Garman-Klass (1980)\* (OHLC Drift-Free):

    $$\\sigma\_{GK} \= \\sqrt{\\frac{252}{N} \\sum\_{t=1}^N \\left\[ 0.5 \\left(\\ln \\frac{H\_t}{L\_t}\\right)^2 \- (2\\ln 2 \- 1\) \\left(\\ln \\frac{C\_t}{O\_t}\\right)^2 \\right\]}$$

  \- \*Yang-Zhang (2000)\* (Minimum Variance Unbiased Estimator with Overnight Jumps):

    $$\\sigma\_{YZ}^2 \= \\sigma\_{\\text{overnight}}^2 \+ k \\cdot \\sigma\_{\\text{open-to-close}}^2 \+ (1-k) \\cdot \\sigma\_{RS}^2$$

    $$k \= \\frac{0.34}{1.34 \+ \\frac{N+1}{N-1}}, \\quad \\sigma\_{RS}^2 \= \\frac{252}{N}\\sum\_{t=1}^N \\left\[ \\ln\\frac{H\_t}{C\_t}\\ln\\frac{H\_t}{O\_t} \+ \\ln\\frac{L\_t}{C\_t}\\ln\\frac{L\_t}{O\_t} \\right\]$$

\- \*\*Gaussian Hidden Markov Model (HMM) 3-State Regime Classifier\*\*:

  \- State $S\_t \\in \\{0: \\text{Bear}, 1: \\text{Neutral}, 2: \\text{Bull}\\}$

  \- Forward-Backward algorithm for smoothed posterior state probabilities $\\gamma\_t(k) \= P(S\_t \= k \\mid R\_{1:T})$.

  \- Baum-Welch (EM) parameter estimation: $\\hat{\\mu}\_k \= \\frac{\\sum \\gamma\_t(k) R\_t}{\\sum \\gamma\_t(k)}$, $\\hat{\\sigma}\_k^2 \= \\frac{\\sum \\gamma\_t(k)(R\_t \- \\hat{\\mu}\_k)^2}{\\sum \\gamma\_t(k)}$.

  \- Expected Regime Duration: $E\[D\_k\] \= \\frac{1}{1 \- P\_{kk}}$.

&nbsp;

\#\#\# 2\. How These Are Used

1\. Feed raw tick or daily OHLCV bars.

2\. Compute Yang-Zhang and Garman-Klass rolling volatilities across multiple horizons (10d, 21d, 63d, 126d, 252d) to construct a \*\*Volatility Cone\*\*.

3\. Feed normalized returns into the 3-state Gaussian HMM to decode the latent market regime via the Viterbi algorithm.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Single Stocks, Broad Equities ETFs (SPY, QQQ, IWM), Index Futures (ES, NQ).

\- \*\*Buy / Long Trigger\*\*: When HMM transitions to State 2 (Bull: $\\mu \> 0, \\sigma \< \\text{median}$) AND current realized Yang-Zhang volatility is at the 15th percentile of its Volatility Cone $\\implies$ enter leveraged long equity positions.

\- \*\*Sell / Short Trigger\*\*: When HMM transitions to State 0 (Bear: $\\mu \< 0, \\sigma \> \\text{median}$) AND realized volatility breaks above the 80th percentile of the Volatility Cone $\\implies$ initiate tactical shorts or liquidate equity holdings to cash/T-bills.

\- \*\*Dynamic Leverage Scaling\*\*: Scale gross leverage proportionally to inverse Yang-Zhang volatility: $\\text{Leverage}\_t \= \\min\\left(2.0, \\frac{\\sigma\_{\\text{target}}}{\\sigma\_{\\text{YZ}, t}}\\right)$.

&nbsp;

\---

&nbsp;

\#\# Track 2: Quantitative Risk & Portfolio Theory (Projects 06–10)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Cornish-Fisher Modified Value at Risk (VaR)\*\*:

  $$\\tilde{z}\_\\alpha \= z\_\\alpha \+ \\frac{1}{6}(z\_\\alpha^2 \- 1)S \+ \\frac{1}{24}(z\_\\alpha^3 \- 3z\_\\alpha)K \- \\frac{1}{36}(2z\_\\alpha^3 \- 5z\_\\alpha)S^2$$

  $$\\text{VaR}\_{\\alpha, \\text{CF}} \= \-(\\mu \- \\tilde{z}\_\\alpha \\sigma)$$

\- \*\*Expected Shortfall (CVaR)\*\*:

  $$\\text{CVaR}\_\\alpha \= \-\\mathbb{E}\[R \\mid R \\le \-\\text{VaR}\_\\alpha\] \= \-\\frac{1}{|R\_t \\le \-\\text{VaR}\_\\alpha|} \\sum\_{R\_t \\le \-\\text{VaR}\_\\alpha} R\_t$$

\- \*\*Markowitz Modern Portfolio Theory (MPT) Quadratic Optimization\*\*:

  $$\\max\_w \\frac{w^T \\mu \- R\_f}{\\sqrt{w^T \\Sigma w}} \\quad \\text{s.t.} \\quad \\sum\_{i=1}^N w\_i \= 1, \\quad l\_i \\le w\_i \\le u\_i$$

\- \*\*Regulatory Exception Backtesting (Kupiec POF Test)\*\*:

  $$LR\_{\\text{POF}} \= \-2 \\ln\\left(\\frac{(1 \- p\_0)^{T-x} p\_0^x}{(1 \- \\hat{p})^{T-x} \\hat{p}^x}\\right) \\sim \\chi^2(1), \\quad \\hat{p} \= \\frac{x}{T}$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Estimate expected returns $\\mu$ and asset covariance $\\Sigma$ with Ledoit-Wolf shrinkage.

2\. Solve quadratic programming with SLSQP to trace the continuous Efficient Frontier.

3\. Compute portfolio Cornish-Fisher VaR and CVaR at 95% and 99% confidence levels.

4\. Backtest model exceedances using the Kupiec Likelihood Ratio test against Basel traffic light boundaries.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Multi-Asset Portfolios (Global Equities, Sovereign Bonds, Gold, Commodities).

\- \*\*Portfolio Rebalancing Trigger\*\*: Rebalance asset weights to the Maximum Sharpe Ratio Tangency Portfolio monthly or when portfolio tracking error exceeds 150 bps.

\- \*\*Risk Gating (Position De-risking)\*\*: If the 1-day 99% Cornish-Fisher VaR breaches the fund risk tolerance (e.g. $\> 2.5\\%$ of NAV) $\\implies$ dynamically de-gross the portfolio by selling high-beta assets and moving capital to short-term T-bills.

\- \*\*CVaR Tail-Hedge Allocation\*\*: If Expected Shortfall deviates upward from Gaussian parametric VaR by $\> 35\\%$ (signaling fat-tail jump risk) $\\implies$ allocate 1–2% of capital to out-of-the-money put options on SPY/QQQ.

&nbsp;

\---

&nbsp;

\#\# Track 3: Systematic Trading Strategies & Backtesting (Projects 11–15)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Bollinger Bands & Mean Reversion Z-Score\*\*:

  $$Z\_t \= \\frac{P\_t \- \\text{SMA}\_t(N)}{\\sigma\_t(N)}, \\quad \\text{Upper}\_t \= \\text{SMA}\_t \+ k\\sigma\_t, \\quad \\text{Lower}\_t \= \\text{SMA}\_t \- k\\sigma\_t$$

\- \*\*Volatility-Targeted Trend Following\*\*:

  $$R\_{12-1, t} \= \\frac{P\_{t-21}}{P\_{t-252}} \- 1, \\quad \\text{Signal}\_t \= \\text{sign}(R\_{12-1, t}), \\quad w\_t \= \\text{Signal}\_t \\cdot \\min\\left(\\frac{\\sigma\_{\\text{target}}}{\\hat{\\sigma}\_t}, L\_{\\text{max}}\\right)$$

\- \*\*Kalman Filter Dynamic Pairs Trading (State-Space Formulation)\*\*:

  \- State: $\\theta\_t \= \[\\alpha\_t, \\beta\_t\]^T$, $\\theta\_t \= \\theta\_{t-1} \+ w\_t, w\_t \\sim \\mathcal{N}(0, Q\_t)$

  \- Measurement: $y\_t \= \[1, x\_t\] \\theta\_t \+ v\_t, v\_t \\sim \\mathcal{N}(0, R)$

  \- Innovation & Gain: $e\_t \= y\_t \- \[1, x\_t\] \\hat{\\theta}\_{t|t-1}, K\_t \= P\_{t|t-1} H\_t^T / S\_t$

  \- Standardized Spread Z-Score: $Z\_t \= \\frac{e\_t}{\\sqrt{S\_t}}$

  \- Ornstein-Uhlenbeck Mean Reversion Half-Life: $dS\_t \= \\kappa(\\mu \- S\_t)dt \+ \\sigma dW\_t \\implies t\_{1/2} \= \\frac{\\ln 2}{\\kappa}$.

\- \*\*Risk Parity (Equal Risk Contribution)\*\*:

  $$\\min\_w \\frac{1}{2} w^T \\Sigma w \- \\frac{1}{N} \\sum\_{i=1}^N \\ln(w\_i) \\implies w\_i (\\Sigma w)\_i \= \\frac{1}{N} \\sigma\_{\\text{portfolio}}^2$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Screen pairs of assets for Engle-Granger cointegration ($p \< 0.05$).

2\. Apply the recursive Kalman filter on tick or daily close data to update the hedge ratio $\\beta\_t$ in real time.

3\. Compute the normalized spread Z-score $Z\_t$ and evaluate the half-life of mean reversion.

4\. Execute strictly lagged portfolio weights ($w\_{t-1}^T R\_t$) with linear fees and borrow costs.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Liquid Equities (KO vs PEP, XOM vs CVX), Commodity ETFs (GLD vs SLV), FX Pairs.

\- \*\*Statistical Arbitrage Buy/Sell Trigger\*\*:

  \- When Kalman spread $Z\_t \\le \-2.0$ (Spread undervalued): \*\*BUY Asset Y\*\* and \*\*SELL $\\beta\_t$ units of Asset X\*\*. Dollar-neutral weights: $w\_y \= \\frac{1}{1 \+ |\\beta\_t|}, w\_x \= \-\\frac{\\beta\_t}{1 \+ |\\beta\_t|}$.

  \- When Kalman spread $Z\_t \\ge \+2.0$ (Spread overvalued): \*\*SELL Asset Y\*\* and \*\*BUY $\\beta\_t$ units of Asset X\*\*.

  \- \*\*Exit Rule\*\*: Liquidate spread position when $|Z\_t| \\le 0.25$ or when holding duration exceeds $2.5 \\times t\_{1/2}$.

  \- \*\*Stop Loss\*\*: Emergency exit if $|Z\_t| \\ge 3.8$ (structural cointegration breakdown).

&nbsp;

\---

&nbsp;

\#\# Track 4: Derivatives & Options Pricing Models (Projects 16–20)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Black-Scholes-Merton European Analytical Solution & Greeks\*\*:

  $$C \= S\_0 e^{-qT} N(d\_1) \- K e^{-rT} N(d\_2), \\quad P \= K e^{-rT} N(-d\_2) \- S\_0 e^{-qT} N(-d\_1)$$

  $$\\Delta \= e^{-qT} N(d\_1), \\quad \\Gamma \= \\frac{e^{-qT}\\phi(d\_1)}{S\_0 \\sigma \\sqrt{T}}, \\quad \\nu \= S\_0 e^{-qT} \\sqrt{T}\\phi(d\_1), \\quad \\Theta \= \-\\frac{S\_0 e^{-qT}\\phi(d\_1)\\sigma}{2\\sqrt{T}} \- r K e^{-rT}N(d\_2)$$

\- \*\*Gatheral Raw SVI (Stochastic Volatility Inspired) Total Variance\*\*:

  $$w(k) \= \\sigma^2(k) T \= a \+ b\\left(\\rho(k \- m) \+ \\sqrt{(k \- m)^2 \+ \\sigma\_{\\text{SVI}}^2}\\right), \\quad k \= \\ln(K / F)$$

\- \*\*Heston (1993) Continuous Stochastic Volatility\*\*:

  $$dS\_t \= (r \- q)S\_t dt \+ \\sqrt{v\_t} S\_t dW\_t^S, \\quad dv\_t \= \\kappa(\\theta \- v\_t)dt \+ \\xi \\sqrt{v\_t} dW\_t^v, \\quad \\mathbb{E}\[dW\_t^S dW\_t^v\] \= \\rho dt$$

  \- Albrecher et al. (2007) stable characteristic function $\\phi(u; T)$ preventing branch cut discontinuities.

  \- Fang-Oosterlee (2008) Fourier-Cosine (COS) series expansion for sub-millisecond pricing:

    $$V(S\_0, K, T) \= K e^{-rT} \\sum\_{k=0}^{N-1} \\sideset{}{'}\\sum \\text{Re}\\left\\{\\phi\\left(\\frac{k\\pi}{b-a}\\right) e^{-i \\frac{k\\pi a}{b-a}}\\right\\} H\_k$$

\- \*\*Longstaff-Schwartz Least Squares Monte Carlo (LSM)\*\*:

  Estimates continuation values for American options via cross-sectional polynomial regression on in-the-money paths: $\\mathbb{E}\[C\_t \\mid S\_t\] \\approx \\sum\_{m=0}^M \\beta\_m L\_m(S\_t)$.

&nbsp;

\#\#\# 2\. How These Are Used

1\. Ingest option chains (strikes, bids, asks, tenors).

2\. Invert market prices to Implied Volatility via Newton-Raphson / Brent.

3\. Fit Gatheral SVI to generate an arbitrage-free volatility surface across all strikes and expirations.

4\. Calibrate Heston parameters $\\{v\_0, \\kappa, \\theta, \\xi, \\rho\\}$ via SLSQP subject to the Feller condition ($2\\kappa\\theta \> \\xi^2$).

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Index Options (SPX, NDX), Single Stock Options, Volatility Indices (VIX).

\- \*\*Volatility Arbitrage (Vega Neutral, Long Gamma)\*\*: When market implied volatility $\\sigma\_{\\text{IV}}$ is significantly below GARCH/Yang-Zhang forecast realized volatility $\\sigma\_{\\text{realized}}$ ($\\sigma\_{\\text{realized}} \- \\sigma\_{\\text{IV}} \> 3.0\\text{ vol points}$):

  \- \*\*BUY ATM Straddles\*\* (Buy Call \+ Buy Put).

  \- Dynamically delta-hedge the underlying equity daily: $\\Delta \\text{Shares} \= \-\\sum \\Delta\_i$.

  \- Profit is generated through Gamma Scalping as the underlying moves: $\\text{P\&L} \\approx \\frac{1}{2} \\Gamma (\\Delta S)^2 \- \\Theta \\Delta t$.

\- \*\*SVI Skew Arbitrage\*\*: When out-of-the-money put implied volatility deviates above the calibrated SVI smile curve $\\implies$ sell overpriced 25-delta puts and buy 10-delta crash puts (put spread collar).

&nbsp;

\---

&nbsp;

\#\# Track 5: Advanced Quantitative Research & Machine Learning (Projects 21–25)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*GJR-GARCH(1,1) Asymmetric Volatility Forecasting\*\*:

  $$\\sigma\_t^2 \= \\omega \+ \\left(\\alpha \+ \\gamma I\_{\\{\\epsilon\_{t-1} \< 0\\}}\\right) \\epsilon\_{t-1}^2 \+ \\beta \\sigma\_{t-1}^2$$

  \- Persistence: $P \= \\alpha \+ \\beta \+ \\frac{1}{2}\\gamma \< 1$

  \- Long-term variance: $\\sigma\_L^2 \= \\frac{\\omega}{1 \- P}$

  \- Multi-step ahead forecast: $\\mathbb{E}\_t\[\\sigma\_{t+h}^2\] \= \\sigma\_L^2 \+ P^{h-1}(\\sigma\_{t+1}^2 \- \\sigma\_L^2)$

\- \*\*Nelson-Siegel & Nelson-Siegel-Svensson Term Structure\*\*:

  $$y(\\tau) \= \\beta\_0 \+ \\beta\_1 \\left(\\frac{1 \- e^{-\\tau/\\lambda}}{\\tau/\\lambda}\\right) \+ \\beta\_2 \\left(\\frac{1 \- e^{-\\tau/\\lambda}}{\\tau/\\lambda} \- e^{-\\tau/\\lambda}\\right)$$

  \- $\\beta\_0$: Level, $\\beta\_1$: Slope, $\\beta\_2$: Curvature. Instantaneous forward rate: $f(\\tau) \= y(\\tau) \+ \\tau y'(\\tau)$.

\- \*\*Fixed-Width Window Fractional Differencing (FFD)\*\*:

  $$(1 \- B)^d \= \\sum\_{k=0}^\\infty w\_k B^k, \\quad w\_0 \= 1, \\quad w\_k \= \-w\_{k-1} \\frac{d \- k \+ 1}{k}$$

  Finds minimum $d^\* \\in (0, 1)$ such that the series achieves ADF stationarity ($p \< 0.01$) while retaining $\>90\\%$ correlation with raw log-prices.

\- \*\*Factor Neutralization / Orthogonalization\*\*:

  $$S\_{\\text{neutral}} \= S \- X (X^T X)^{-1} X^T S$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Apply FFD to raw financial features to preserve memory while passing stationarity tests.

2\. Cross-validate using Purged & Embargoed TimeSeries splits to eliminate serial leakage.

3\. Fit GJR-GARCH on return residuals to dynamically forecast volatility term structures.

4\. Fit Nelson-Siegel curves on sovereign yield matrices to extract Level, Slope, and Curvature factors.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: US Treasury Yields, Fixed Income ETFs (SHY, IEF, TLT), Equity ML Portfolios.

\- \*\*Yield Curve Butterfly Spread Trading\*\*: When Nelson-Siegel Curvature factor $\\beta\_2$ reaches the 95th historical percentile (curve is abnormally humped at the belly):

  \- \*\*BUY 2Y Treasury Notes\*\* and \*\*BUY 10Y Treasury Bonds\*\* (Wings).

  \- \*\*SELL 5Y Treasury Notes\*\* (Belly) in duration-neutral weights ($D\_{\\text{wing1}} w\_1 \+ D\_{\\text{wing2}} w\_2 \= D\_{\\text{belly}} w\_{\\text{belly}}$).

  \- Profit from the mean reversion of curvature back toward the historical mean.

\- \*\*Fractional Differenced ML Alpha\*\*: When regularized Ridge model predicts forward return $\> \+50\\text{ bps}$ on an FFD feature vector $\\implies$ take a long position with size scaled by inverse GJR-GARCH volatility forecast.

&nbsp;

\---

&nbsp;

\#\# Track 6: Market Microstructure & Production Execution Systems (Projects 26–30)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Microstructure Indicators\*\*:

  \- Order Book Imbalance: $I \= \\frac{V\_b \- V\_a}{V\_b \+ V\_a} \\in \[-1, \+1\]$

  \- Volume-Weighted Micro-Price: $P\_{\\text{micro}} \= \\frac{V\_b P\_a \+ V\_a P\_b}{V\_b \+ V\_a}$

\- \*\*Almgren-Chriss (2000) Optimal Liquidation Trajectory\*\*:

  Minimizes implementation shortfall cost and holding risk for $X\_0$ shares over $N$ intervals ($\\tau \= T/N$):

  $$x\_j \= \\frac{\\sinh(\\kappa (T \- t\_j))}{\\sinh(\\kappa T)} X\_0, \\quad \\kappa \= \\sqrt{\\frac{\\lambda \\sigma^2}{\\eta}}$$

  $$\\mathbb{E}\[\\text{Cost}\] \= \\frac{1}{2}\\gamma X\_0^2 \+ \\eta \\sum\_{j=1}^N \\frac{(x\_{j-1} \- x\_j)^2}{\\tau}, \\quad V\[\\text{Cost}\] \= \\sigma^2 \\tau \\sum\_{j=1}^N x\_j^2$$

\- \*\*Volume Synchronized Probability of Toxicity (VPIN)\*\*:

  $$V\_\\tau^B \= V \\cdot \\Phi\\left(\\frac{P\_\\tau \- P\_{\\tau-1}}{\\sigma\_{\\Delta P}}\\right), \\quad V\_\\tau^S \= V \- V\_\\tau^B, \\quad \\text{VPIN} \= \\frac{\\sum\_{\\tau=1}^N |V\_\\tau^B \- V\_\\tau^S|}{N \\cdot V}$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Maintain an active Level 2 Limit Order Book with FIFO price-time queues.

2\. Ingest trade prints into constant-volume buckets ($V \= \\text{ADV} / N\_{\\text{buckets}}$) to compute rolling VPIN.

3\. Schedule block liquidations via Almgren-Chriss hyperbolic trajectories.

4\. Execute via Smart Order Router (SOR) switching between TWAP, VWAP, and Iceberg algorithms.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: High-Frequency Equities, Single Stocks, Index Futures.

\- \*\*Short-Term Micro-Price Momentum Trigger\*\*:

  \- When $P\_{\\text{micro}} \- P\_{\\text{mid}} \> \+0.5 \\times \\text{Spread}$ AND Order Book Imbalance $I \> \+0.40$ $\\implies$ submit aggressive limit buy orders at $P\_{\\text{bid}} \+ 1\\text{ tick}$, capturing short-term fill momentum.

\- \*\*Toxic Flow Avoidance (Circuit Breaker)\*\*:

  \- When real-time $\\text{VPIN} \> 0.85$ (99th historical percentile) $\\implies$ cancel all passive quoting limit orders immediately, widen bid-ask spreads by $3\\times$, and pause market making to avoid adverse selection before a flash crash.

\- \*\*Execution Cost Reduction\*\*:

  \- Slice a 500,000 share rebalance order into Almgren-Chriss intervals to save 15–30 basis points in market impact costs versus naive market orders.

&nbsp;

\---

&nbsp;

\#\# Track 7: Frontier Quant AI, Advanced Math & Alternative Data (Projects 31–35)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*SEC 10-K Semantic Drift & Loughran-McDonald Sentiment\*\*:

  $$\\text{Drift}\_t \= 1 \- \\frac{\\mathbf{v}\_t \\cdot \\mathbf{v}\_{t-1}}{\\|\\mathbf{v}\_t\\|\_2 \\|\\mathbf{v}\_{t-1}\\|\_2} \\in \[0, 2\], \\quad \\text{Sentiment} \= \\frac{N\_{\\text{pos}} \- N\_{\\text{neg}}}{N\_{\\text{pos}} \+ N\_{\\text{neg}} \+ \\epsilon}$$

\- \*\*Supply-Chain Graph Convolutional Network (GCN)\*\*:

  $$H^{(l+1)} \= \\text{ReLU}\\left(\\tilde{D}^{-1/2} \\tilde{A} \\tilde{D}^{-1/2} H^{(l)} W^{(l)}\\right), \\quad \\tilde{A} \= A \+ I\_N, \\quad \\tilde{D}\_{ii} \= \\sum\_j \\tilde{A}\_{ij}$$

  Propagates customer earnings surprises downstream along directed revenue dependency matrix $A\_{ij}$.

\- \*\*Wasserstein Distributionally Robust Portfolio Optimization (DRO)\*\*:

  $$\\min\_{w \\in \\mathcal{W}} \\max\_{\\mathbb{Q}: W\_1(\\mathbb{Q}, \\hat{\\mathbb{P}}\_N) \\le \\epsilon} \\mathbb{E}\_{\\mathbb{Q}}\\left\[-w^T \\xi \+ \\frac{\\gamma}{2}(w^T \\xi \- w^T \\hat{\\mu})^2\\right\]$$

  \*\*Convex Dual Reformulation\*\*:

  $$\\min\_{w \\in \\mathcal{W}} \\left( \-w^T \\hat{\\mu} \+ \\frac{\\gamma}{2} w^T \\hat{\\Sigma} w \+ \\epsilon \\|w\\|\_2 \\right) \\quad \\text{s.t.} \\quad \\sum\_{i=1}^N w\_i \= 1, \\quad w\_i \\ge 0$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Parse annual SEC 10-K filings; compute cosine drift on Item 1A (Risk Factors) and Item 7 (MD\&A).

2\. Construct economic supply-chain graphs from customer disclosure filings.

3\. Solve Wasserstein DRO convex quadratic program across ambiguity radius $\\epsilon$.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Russell 3000 Equities, Global Supply Chain Networks.

\- \*\*The "Lazy Prices" Alpha Strategy (Cohen, Malloy, Nguyen 2020)\*\*:

  \- \*\*BUY / LONG\*\*: Quintile 1 (Low-Drift / "Lazy" Disclosers: firms making virtually no changes to their Risk Factors text, signaling stable operations).

  \- \*\*SELL / SHORT\*\*: Quintile 5 (High-Drift Disclosers: firms heavily modifying their Risk Factors with unpriced negative legal/operational revisions).

  \- Hold for 3 to 12 months in a dollar-neutral portfolio; historically generates $+6\\%$ to $+10\\%$ annualized alpha.

\- \*\*Supply-Chain Customer Spillover Momentum\*\*:

  \- When a major customer (e.g. Apple, Boeing) beats earnings by $\> 10\\%$ and rallies $\> 4\\%$ on announcement day $\\implies$ \*\*BUY key suppliers\*\* (firms deriving $\> 25\\%$ revenue from that customer) on day $t+1$. Capture the 2–10 day delayed price reaction.

\- \*\*Wasserstein DRO Allocation\*\*: Allocate capital using $\\epsilon \= 0.015$ shrinkage to prevent portfolio concentration in in-sample winners that collapse out-of-sample.

&nbsp;

\---

&nbsp;

\#\# Track 8: Climate Quantitative Finance & Carbon Markets (Projects 36–40)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Clean Spreads & Fuel-Switching Carbon Parity\*\*:

  $$\\text{CSS} \= P\_{\\text{power}} \- \\frac{P\_{\\text{gas}}}{\\eta\_{\\text{gas}}} \- \\text{EF}\_{\\text{gas}} \\cdot P\_{\\text{carbon}}$$

  $$\\text{CDS} \= P\_{\\text{power}} \- \\frac{P\_{\\text{coal}}}{\\eta\_{\\text{coal}}} \- \\text{EF}\_{\\text{coal}} \\cdot P\_{\\text{carbon}}$$

  $$P\_{\\text{switch}} \= \\frac{\\frac{P\_{\\text{gas}}}{\\eta\_{\\text{gas}}} \- \\frac{P\_{\\text{coal}}}{\\eta\_{\\text{coal}}}}{\\text{EF}\_{\\text{coal}} \- \\text{EF}\_{\\text{gas}}}$$

\- \*\*Green Bond Greenium Decomposition\*\*:

  $$\\text{Greenium}\_t \= (y\_{\\text{vanilla}, t} \- y\_{\\text{green}, t}) \\times 10^4 \\quad (\\text{in bps})$$

  $$\\text{Pure Greenium} \= \\text{Raw Spread} \- \\Delta y\_{\\text{maturity}} \- \\Delta y\_{\\text{duration}} \+ \\Delta y\_{\\text{liquidity}}$$

\- \*\*NGFS Scenario Discounted Cash Flow Climate VaR\*\*:

  $$\\Delta \\text{EBITDA}\_i(t) \= \-\\text{CarbonPrice}(t) \\times (\\text{Scope 1}\_i \+ \\text{Scope 2}\_i \+ \\alpha \\text{Scope 3}\_i) \\times (1 \- \\beta\_{\\text{pass}, i})$$

  $$\\text{Climate VaR}\_i \= \\frac{\\text{Stressed Equity}\_i \- \\text{Market Cap}\_i}{\\text{Market Cap}\_i} \\in \[-100\\%, 0\\%\]$$

\- \*\*Renewable PPA Duck Curve Cannibalization\*\*:

  $$\\text{Capture Price} \= \\frac{\\sum\_{t=1}^T P\_t \\times Q\_t}{\\sum\_{t=1}^T Q\_t}, \\quad \\text{Capture Rate} \= \\frac{\\text{Capture Price}}{\\bar{P}\_{\\text{baseload}}}$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Track real-time power, gas TTF, coal ARA, and EU ETS EUA carbon allowance prices.

2\. Solve the fuel-switching equation to determine which thermal generation technology sets the power price.

3\. Stress test corporate portfolios under NGFS Net Zero 2050 shadow carbon prices ($140/t).

4\. Simulate hourly solar/wind generation profiles against wholesale electricity spot prices.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: EU ETS Carbon Allowance Futures (EUA), European Power Futures, Green Bonds, Utility Equities.

\- \*\*EUA Carbon Allowance Mean-Reversion Arbitrage\*\*:

  \- When market carbon price $P\_{\\text{carbon}} \< P\_{\\text{switch}} \- €5.00$ $\\implies$ coal is more profitable than gas $\\implies$ utilities must burn coal $\\implies$ coal carbon emissions are $2.5\\times$ higher $\\implies$ compliance carbon allowance demand will surge.

  \- \*\*BUY EUA Carbon Futures contracts\*\*; hold until $P\_{\\text{carbon}}$ converges to $P\_{\\text{switch}}$.

\- \*\*Green Bond Relative-Value Arbitrage\*\*:

  \- If a corporate Green Bond trades with an abnormally wide greenium ($\> 15\\text{ bps}$ after duration/liquidity adjustment) $\\implies$ \*\*SELL the Green Bond\*\* and \*\*BUY the conventional twin bond\*\*, locking in mean reversion.

\- \*\*Satellite Plume Decarbonization Long/Short\*\*:

  \- \*\*SHORT\*\* industrial emitters with large unannounced methane/CO2 satellite plumes (high regulatory liability and impending fines).

  \- \*\*LONG\*\* verified decarbonization leaders.

&nbsp;

\---

&nbsp;

\#\# Track 9: Decentralized Finance (DeFi), AMM Liquidity & Crypto Quant (Projects 41–45)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Uniswap v3 Concentrated Liquidity Virtual Reserves\*\*:

  $$\\left(x \+ \\frac{L}{\\sqrt{P\_b}}\\right)\\left(y \+ L\\sqrt{P\_a}\\right) \= L^2, \\quad P(i) \= 1.0001^i$$

  $$\\text{Capital Efficiency Multiplier} \= \\frac{1}{1 \- (P\_a / P\_b)^{1/4}}$$

\- \*\*Loss-Versus-Rebalancing (LVR) Adverse Selection\*\*:

  $$\\text{LVR}\_t \= \\int\_0^t \\frac{\\sigma^2}{8} S\_u L\_u du, \\quad \\sigma\_{\\text{breakeven}} \= \\sqrt{\\frac{8 \\times \\text{Fees}}{\\int\_0^T S\_t L\_t dt}}$$

\- \*\*Optimal Flash Loan Spatial Arbitrage Sizing ($\\Delta x^\*$CPMM)\*\*:

  $$\\Delta x^\* \= \\frac{\\sqrt{x\_1 x\_2 y\_1 y\_2 \\gamma\_1 \\gamma\_2} \- x\_1 y\_2}{\\gamma\_1 y\_2 \+ \\gamma\_1 \\gamma\_2 y\_1}$$

\- \*\*Perpetual Futures 8h Funding Rate Clamping\*\*:

  $$\\text{Funding Rate}\_t \= \\text{Clamp}\\left( \\text{Premium Index}\_t \+ \\text{Clamp}(\\text{Interest} \- \\text{Premium}, \-0.05\\%, \+0.05\\%), \-0.75\\%, \+0.75\\% \\right)$$

\- \*\*On-Chain MVRV Z-Score\*\*:

  $$\\text{MVRV} \= \\frac{\\text{Market Cap}}{\\text{Realized Cap}}, \\quad Z\_{\\text{MVRV}} \= \\frac{\\text{Market Cap} \- \\text{Realized Cap}}{\\sigma(\\text{Market Cap})}$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Monitor on-chain pool reserves across Uniswap v2/v3, Sushiswap, and Curve.

2\. Calculate LVR to assess whether liquidity provider fees exceed adverse selection losses.

3\. Screen centralized and decentralized perpetual futures for funding rate dislocations.

4\. Calculate MVRV Z-scores to identify macro cycle extremes.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Spot Crypto (BTC, ETH, SOL), Perpetual Futures, Uniswap v3 LP positions.

\- \*\*Delta-Neutral Perpetual Funding Rate Cash-and-Carry\*\*:

  \- When 8-hour perpetual funding rate annualized exceeds $+15\\%\\text{ APY}$:

  \- \*\*BUY Spot ETH\*\* (or staked stETH for $+3.5\\%$ additional yield).

  \- \*\*SELL / SHORT ETH Perpetual Futures\*\* in equal dollar notional.

  \- Collect 8-hour funding rate payments with zero directional price risk.

  \- Rebalance when funding rate drops below $3.0\\%$ annualized.

\- \*\*LVR-Aware Dynamic AMM Liquidity Provision\*\*:

  \- Only deposit liquidity into a Uniswap v3 pool if historical asset volatility $\\sigma \< \\sigma\_{\\text{breakeven}}$ (guaranteeing fee revenue exceeds arbitrageur LVR drain).

\- \*\*Macro Cycle On-Chain Accumulation / Distribution\*\*:

  \- When on-chain $Z\_{\\text{MVRV}} \< 0.10$ AND Net Exchange Flow is negative (whales moving coins to cold storage) $\\implies$ \*\*ACCUMULATE / BUY Spot BTC\*\*.

  \- When $Z\_{\\text{MVRV}} \> 3.20$ AND Exchange Inflows surge $\\implies$ \*\*DISTRIBUTE / SELL Spot BTC\*\* into stablecoins.

&nbsp;

\---

&nbsp;

\#\# Track 10: Global Macro AI, Crypto & Cross-Economy Sentiment (Projects 46–50)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Central Bank Monetary Policy Hawk/Dove Index\*\*:

  $$H\_t \= \\frac{N\_{\\text{hawkish}} \- N\_{\\text{dovish}}}{N\_{\\text{hawkish}} \+ N\_{\\text{dovish}} \+ \\epsilon} \\in \[-1.0, \+1.0\]$$

  Taylor Rule Gap: $\\Delta i\_t \= \\alpha \+ \\beta\_1 (\\pi\_t \- \\pi^\*) \+ \\beta\_2 (y\_t \- y^\*) \+ \\gamma H\_t$.

\- \*\*Diebold-Yilmaz (2012) Sovereign Contagion Spillovers\*\*:

  $$\\theta\_{ij}(H) \= \\frac{\\sigma\_{jj}^{-1} \\sum\_{h=0}^{H-1} (e\_i' A\_h \\Sigma e\_j)^2}{\\sum\_{h=0}^{H-1} (e\_i' A\_h \\Sigma A\_h' e\_i)}, \\quad \\text{Total Spillover Index} \= \\frac{\\sum\_{i \\ne j} \\tilde{\\theta}\_{ij}(H)}{N} \\times 100\\%$$

\- \*\*Covered (CIP) & Uncovered (UIP) Interest Rate Parity\*\*:

  $$\\text{CIP Basis (bps)} \= \\left( \\frac{F\_{t, T}}{S\_t} (1 \+ r\_{\\text{foreign}} T) \- (1 \+ r\_{\\text{domestic}} T) \\right) \\times 10^4$$

\- \*\*Malz (1997) FX Volatility Surface\*\*:

  $$\\sigma(\\Delta) \= \\sigma\_{\\text{ATM}} \- 2 \\cdot RR\_{25} (\\Delta \- 0.5) \+ 16 \\cdot BF\_{25} (\\Delta \- 0.5)^2$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Parse multilingual press releases and meeting minutes across Fed, ECB, BOJ, RBI, BCB, Banxico, and PBOC.

2\. Compute directional sovereign debt volatility spillover matrices from VAR models.

3\. Track 25-delta Risk Reversal ($RR\_{25}$) and Butterfly ($BF\_{25}$) FX options quotes.

4\. Run an autonomous 4-agent hedge fund committee (Macro, Crypto, Sentiment, PM) with Black-Litterman blending.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: G10 & Emerging Market FX Pairs (USD, EUR, JPY, BRL, MXN, INR), Sovereign CDS.

\- \*\*Cross-Economy FX Carry Trade\*\*:

  \- Rank currencies by short-term central bank interest rates.

  \- \*\*BUY / LONG High-Yielding Currencies\*\* (e.g. BRL, MXN).

  \- \*\*SELL / SHORT Low-Yielding Funding Currencies\*\* (e.g. JPY, CHF).

  \- \*\*Crash Protection Filter\*\*: Exit or hedge carry exposure when the FX 25-delta Risk Reversal ($RR\_{25}$) spikes above the 90th percentile (signaling impending currency crash).

\- \*\*Central Bank Tone Surprise Trade\*\*:

  \- When Central Bank Hawk/Dove score $H\_t$ diverges positively from Taylor Rule expectation by $\> 0.35$ $\\implies$ \*\*SELL 2Y Sovereign Interest Rate Futures\*\* and \*\*BUY Domestic Currency Spot\*\* within 60 seconds of statement release.

&nbsp;

\---

&nbsp;

\#\# Track 11: Prediction Markets, Hawkes Processes & Statistical Rigor (Projects 51–55)

&nbsp;

\#\#\# 1\. Core Formulae & Algorithms

\- \*\*Prediction Market Parity Arbitrage\*\*:

  \- Intra-Venue Parity: $\\text{Gross Edge} \= 1.0 \- (P\_{\\text{yes}}^{\\text{ask}} \+ P\_{\\text{no}}^{\\text{ask}})$

  \- Cross-Venue Inter-Market Parity: $e\_{\\times} \= 1.0 \- (P\_{\\text{yes}, A}^{\\text{ask}} \+ P\_{\\text{no}, B}^{\\text{ask}})$

  \- Fractional Kelly Staking:

    $$b \= \\frac{1 \- \\bar{p}}{\\bar{p}}, \\quad f^\* \= \\frac{b p \- q}{b}, \\quad f\_{\\text{sized}} \= \\text{fraction} \\times f^\*$$

\- \*\*Multivariate Hawkes Point Process\*\*:

  $$\\lambda\_m(t) \= \\mu\_m \+ \\sum\_{n=0}^{M-1} \\sum\_{t\_j^n \< t} \\alpha\_{mn} e^{-\\beta\_{mn}(t \- t\_j^n)}$$

  \- Information Absorption Half-Life: $t\_{1/2} \= \\frac{\\ln 2}{\\beta\_{10}}$

  \- Endogeneity Ratio: $\\rho(A) \= \\max |\\text{eig}(A)|$, where $A\_{mn} \= \\frac{\\alpha\_{mn}}{\\beta\_{mn}}$

  \- Random Time-Change Theorem: $\\int\_{t\_{k-1}}^{t\_k} \\lambda(s) ds \\stackrel{i.i.d.}{\\sim} \\text{Exp}(1)$.

\- \*\*Implied Risk-Neutral Density (Breeden-Litzenberger 1978)\*\*:

  $$q(K) \= e^{rT} \\frac{\\partial^2 C(K)}{\\partial K^2}$$

\- \*\*Deflated Sharpe Ratio (DSR)\*\* \*(Bailey & López de Prado 2014)\*:

  $$\\mathbb{E}\[\\max\_N \\widehat{SR}\] \\approx \\overline{SR} \+ \\sigma\_{SR} \\left\[ (1 \- \\gamma) \\Phi^{-1}\\left(1 \- \\frac{1}{N\_{\\text{eff}}}\\right) \+ \\gamma \\Phi^{-1}\\left(1 \- \\frac{1}{N\_{\\text{eff}} \\cdot e}\\right) \\right\]$$

  $$\\text{DSR} \= \\text{PSR}\\left(\\mathbb{E}\[\\max\_N \\widehat{SR}\]\\right) \= \\Phi\\left(\\frac{(\\widehat{SR} \- \\mathbb{E}\[\\max\_N \\widehat{SR}\])\\sqrt{T-1}}{\\sqrt{1 \- S \\cdot \\widehat{SR} \+ \\frac{K-1}{4}\\widehat{SR}^2}}\\right) \\in \[0, 1\]$$

\- \*\*Avellaneda-Stoikov (2008) High-Frequency Quoting\*\*:

  $$r(s, q, t) \= s \- q \\gamma \\sigma^2 (T \- t), \\quad \\delta^a \+ \\delta^b \= \\gamma \\sigma^2 (T \- t) \+ \\frac{2}{\\gamma} \\ln\\left(1 \+ \\frac{\\gamma}{\\kappa}\\right)$$

&nbsp;

\#\#\# 2\. How These Are Used

1\. Stream live order books from Polymarket and Kalshi; evaluate intra- and cross-venue parity.

2\. Fit bivariate Hawkes processes to macroeconomic news and high-frequency trade spikes.

3\. Compute DSR to audit backtests against selection bias across $N$ explored trials.

4\. Update Avellaneda-Stoikov reservation prices dynamically based on live inventory.

&nbsp;

\#\#\# 3\. Trading Applications & Buy/Sell Decisions

\- \*\*Instruments\*\*: Binary Event Contracts (Polymarket, Kalshi), Index Options (SPY), HFT Equities.

\- \*\*Risk-Free Prediction Market Arbitrage\*\*:

  \- When Polymarket YES ask is $\\$0.45$ and Kalshi NO ask is $\\$0.51$:

  \- Sum of asks \= $\\$0.96 \< \\$1.00$.

  \- \*\*BUY YES on Polymarket\*\* and \*\*BUY NO on Kalshi\*\*.

  \- Collect guaranteed $\\$1.00$ at resolution regardless of event outcome, locking in $+4.17\\%$ risk-free return on capital.

\- \*\*Hawkes News Momentum & Absorption Fade\*\*:

  \- Immediately upon macro news arrival ($t\_0$), if news-to-price kernel $\\alpha\_{10}$ is large $\\implies$ execute aggressive market orders in direction of surprise.

  \- After elapsed time $t \> 3 \\times t\_{1/2}$ (information fully absorbed) $\\implies$ initiate mean-reversion fade orders as endogenous self-excitation dissipates.

\- \*\*Avellaneda-Stoikov HFT Market Making\*\*:

  \- When holding positive inventory ($q \> 0$) $\\implies$ lower reservation price $r(s, q, t) \< s$, skewing ask quote lower (attracting buyer fills) and bid quote lower (deterring seller hits) to passively restore zero inventory.

&nbsp;

\---

&nbsp;

\#\# Complete Multi-Asset Summary Reference Matrix

&nbsp;

| Track | Primary Financial Instruments | Core Mathematical Models | Key Buy/Sell Decision Rules |

| :--- | :--- | :--- | :--- |

| \*\*1. Foundations\*\* | Equities, ETFs, Index Futures | Yang-Zhang Volatility, Gaussian HMM (Baum-Welch EM, Viterbi) | Buy leveraged equity when HMM transitions to Bull and vol is at 15th percentile of Vol Cone; exit to cash in Bear regime. |

| \*\*2. Risk Models\*\* | Multi-Asset Portfolios, SPY Puts | Cornish-Fisher VaR, Expected Shortfall CVaR, Markowitz SLSQP | Rebalance to Max Sharpe Tangency portfolio; buy tail-hedge puts when CVaR diverges $\>35\\%$ from Gaussian VaR. |

| \*\*3. Systematic Alpha\*\* | Single Stocks, Pairs (KO/PEP), Futures | Bollinger Z-Score, TSMOM Trend, Online Kalman Filter ($\\beta\_t$) | Buy Asset Y and sell $\\beta\_t$ Asset X when Kalman spread $Z\_t \\le \-2.0$; exit at $|Z\_t| \\le 0.25$; stop loss at $|Z\_t| \\ge 3.8$. |

| \*\*4. Derivatives\*\* | European/American Options, VIX | BSM Greeks, SVI Surface, Heston COS Pricer, Longstaff-Schwartz | Buy ATM Straddles and gamma-scalp daily when realized vol exceeds implied vol by $\>3.0$ points; sell overpriced SVI puts. |

| \*\*5. Advanced ML\*\* | Treasury Yields, Fixed Income ETFs | GJR-GARCH(1,1), Nelson-Siegel, Fractional Differencing (FFD) | Trade duration-neutral Treasury butterfly spreads when NS Curvature $\\beta\_2$ reaches 95th percentile; trade FFD Ridge ML signals. |

| \*\*6. Microstructure\*\* | Equities, Futures, HFT Books | Level 2 LOB, Almgren-Chriss Optimal Slicing, VPIN Toxicity | Buy when Micro-Price $\>$ Mid-Price by $\>0.5\\times$ Spread and OBI $\>+0.40$; cancel passive quoting when VPIN $\>0.85$ (flash crash warning). |

| \*\*7. Frontier AI\*\* | Russell 3000 Equities, Supply Chains | SEC 10-K Cosine Drift, Supply-Chain GNN, Wasserstein DRO | Long Low-Drift ("Lazy") disclosers and Short High-Drift disclosers; buy suppliers following positive customer earnings surprises. |

| \*\*8. Climate Finance\*\* | EU ETS Carbon (EUA), Green Bonds | Clean Spark/Dark Spreads, $P\_{\\text{switch}}$, NGFS DCF Climate VaR | Buy EUA Carbon Futures when $P\_{\\text{carbon}} \< P\_{\\text{switch}} \- €5$; sell green bonds when greenium $\>15\\text{ bps}$ over vanilla twin bonds. |

| \*\*9. DeFi & Crypto\*\* | Spot Crypto, Perp Futures, Uniswap v3 | Concentrated Virtual Reserves, LVR Integral, Perp Funding Basis | Long Spot ETH \+ Short Perp ETH when 8h funding rate $\>15\\%\\text{ APY}$; provide Uniswap v3 liquidity only when vol $\<\\sigma\_{\\text{breakeven}}$. |

| \*\*10. Global Macro\*\* | G10 & EM Currencies (USD, EUR, BRL) | Central Bank NLP Tone, Diebold-Yilmaz GFEVD, Malz FX Surface | Long high-yield EM currencies and Short low-yield funding currencies; exit carry trades when 25-delta Risk Reversal spikes above 90th percentile. |

| \*\*11. Prediction & Rigor\*\* | Polymarket, Kalshi, SPY Options | Kelly Criterion, Bivariate Hawkes Process, Breeden-Litzenberger RND, DSR | Buy YES on Polymarket and NO on Kalshi when sum of asks $\< \\$1.00$ with Kelly stake; reject strategies where Deflated Sharpe Ratio $\<95\\%$. |

&nbsp;

\---

&nbsp;