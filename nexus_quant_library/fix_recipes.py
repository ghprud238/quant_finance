import re

# 1. Update data_engine.py to support arbitrary ticker lengths dynamically
with open("/working_dir/nexus_quant_platform/src/nexus_quant/core/data_engine.py", "r") as f:
    content = f.read()

# Replace correlation matrix generation to scale dynamically with len(tickers)
old_gen = '''        n_tickers = len(tickers)
        if tickers == ["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "SPY"]:
            corr = np.array([
                [1.00, 0.75, 0.70, 0.68, 0.25, 0.85],
                [0.75, 1.00, 0.78, 0.72, 0.22, 0.88],
                [0.70, 0.78, 1.00, 0.74, 0.20, 0.82],
                [0.68, 0.72, 0.74, 1.00, 0.18, 0.80],
                [0.25, 0.22, 0.20, 0.18, 1.00, 0.45],
                [0.85, 0.88, 0.82, 0.80, 0.45, 1.00]
            ])
        else:
            corr = np.eye(n_tickers) * 0.6 + 0.4'''

new_gen = '''        n_tickers = len(tickers)
        if tickers == ["AAPL", "MSFT", "GOOG", "AMZN", "XOM", "SPY"]:
            corr = np.array([
                [1.00, 0.75, 0.70, 0.68, 0.25, 0.85],
                [0.75, 1.00, 0.78, 0.72, 0.22, 0.88],
                [0.70, 0.78, 1.00, 0.74, 0.20, 0.82],
                [0.68, 0.72, 0.74, 1.00, 0.18, 0.80],
                [0.25, 0.22, 0.20, 0.18, 1.00, 0.45],
                [0.85, 0.88, 0.82, 0.80, 0.45, 1.00]
            ])
        else:
            corr = np.full((n_tickers, n_tickers), 0.45)
            np.fill_diagonal(corr, 1.0)'''

if "correlated_z = z @ L.T" in content:
    content = content.replace(old_gen, new_gen)
    with open("/working_dir/nexus_quant_platform/src/nexus_quant/core/data_engine.py", "w") as f:
        f.write(content)

print("[+] data_engine.py updated.")
