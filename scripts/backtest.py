"""Backtest the shipped model.joblib on real AAPL daily closes.

Fetches the last 12 months from Yahoo Finance, predicts each day's close from
the previous day's close with the committed pipeline, and reports MAE / RMSE /
R^2 alongside the naive persistence baseline (tomorrow = today).
"""

import json

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#d5d4d0",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.labelcolor": "#52514e",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
})

BLUE = "#2a78d6"   # series 1: actual
AQUA = "#1baf7a"   # series 2: predicted
TEXT = "#0b0b0b"

model = joblib.load("model.joblib")
df = yf.download("AAPL", period="12mo", interval="1d", progress=False)
closes = df["Close"].to_numpy().ravel()
dates = df.index

lag1 = closes[:-1].reshape(-1, 1)
actual = closes[1:]
pred = model.predict(lag1).ravel()
naive = closes[:-1]  # persistence baseline

def metrics(y, yhat):
    err = y - yhat
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "r2": round(1 - ss_res / ss_tot, 4),
    }

result = {
    "ticker": "AAPL",
    "days": len(actual),
    "start": str(dates[1].date()),
    "end": str(dates[-1].date()),
    "model": metrics(actual, pred),
    "naive_baseline": metrics(actual, naive),
}
print(json.dumps(result, indent=2))

# Plot: last ~6 months for readability
n = min(126, len(actual))
d, a, p = dates[1:][-n:], actual[-n:], pred[-n:]
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
ax.plot(d, a, color=BLUE, lw=2, label="Actual close")
ax.plot(d, p, color=AQUA, lw=2, ls=(0, (4, 2)), label="Predicted (shipped model)")
ax.text(d[-1], a[-1], "  Actual", color=BLUE, fontweight="bold", va="bottom")
ax.text(d[-1], p[-1], "  Predicted", color=AQUA, fontweight="bold", va="top")
ax.set_title(
    f"Shipped model backtest — AAPL next-day close, last {n} trading days\n"
    f"MAE \\${result['model']['mae']} · RMSE \\${result['model']['rmse']} · R² {result['model']['r2']}",
    color=TEXT, fontsize=11, loc="left",
)
ax.grid(axis="y", color="#eceae6", lw=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylabel("Close (USD)")
ax.margins(x=0.02)
ax.legend(frameon=False, loc="upper left", labelcolor="#52514e")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("docs/backtest.png", bbox_inches="tight")
print("wrote docs/backtest.png")
