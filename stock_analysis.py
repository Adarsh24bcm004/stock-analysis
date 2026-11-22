"""
Stock Market Analysis Project
==============================
Goal: Analyse historical stock data and predict whether a stock price
      is likely to go UP or DOWN the next trading day.

Approach: Moving-Average Crossover + RSI signals fed into a simple
          Logistic Regression classifier.

Libraries: pandas, numpy, matplotlib, scikit-learn
Data source: Synthetic / CSV (easily swappable with yfinance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ──────────────────────────────────────────────
# 1.  GENERATE SYNTHETIC STOCK DATA
#     (Replace this section with real data if you
#      have a CSV or use yfinance — see bottom note)
# ──────────────────────────────────────────────
np.random.seed(42)
days = 1000  # ~4 years of trading days

dates = pd.bdate_range(start="2022-01-03", periods=days)
price = 150.0  # starting price
prices = [price]

for _ in range(days - 1):
    change_pct = np.random.normal(loc=0.0003, scale=0.015)
    price *= 1 + change_pct
    prices.append(price)

df = pd.DataFrame(
    {
        "Date": dates,
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 10_000_000, size=days),
    }
)
df.set_index("Date", inplace=True)

# Add Open, High, Low for realism
df["Open"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])
df["High"] = df[["Open", "Close"]].max(axis=1) * (1 + np.random.uniform(0, 0.01, days))
df["Low"] = df[["Open", "Close"]].min(axis=1) * (1 - np.random.uniform(0, 0.01, days))

print("=" * 60)
print("STOCK MARKET DATA ANALYSIS PROJECT")
print("=" * 60)
print(f"\nDataset shape : {df.shape}")
print(f"Date range    : {df.index[0].date()} → {df.index[-1].date()}")
print(f"\nFirst 5 rows:\n{df.head()}\n")

# ──────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ──────────────────────────────────────────────
# 2a. Moving Averages
df["SMA_10"] = df["Close"].rolling(window=10).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# 2b. Relative Strength Index (RSI) — 14-day
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# 2c. Daily return (percentage)
df["Daily_Return"] = df["Close"].pct_change() * 100

# 2d. Volatility — 10-day rolling std of returns
df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

# 2e. Volume change (%)
df["Volume_Change"] = df["Volume"].pct_change() * 100

# 2f. Price momentum (5-day)
df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1

# ──────────────────────────────────────────────
# 3.  CREATE THE TARGET VARIABLE
#     1 = price goes UP tomorrow, 0 = price goes DOWN
# ──────────────────────────────────────────────
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Drop rows with NaN values from rolling calculations
df.dropna(inplace=True)

print(f"Features engineered. Final dataset: {df.shape}")
print(f"Target distribution:\n{df['Target'].value_counts()}\n")

# ──────────────────────────────────────────────
# 4.  VISUALISATIONS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Stock Market Data — Exploratory Analysis", fontsize=16, fontweight="bold")

# 4a. Price + Moving Averages
axes[0].plot(df.index, df["Close"], label="Close Price", linewidth=1)
axes[0].plot(df.index, df["SMA_10"], label="SMA 10", linewidth=1, linestyle="--")
axes[0].plot(df.index, df["SMA_50"], label="SMA 50", linewidth=1, linestyle="--")
axes[0].set_ylabel("Price ($)")
axes[0].legend()
axes[0].set_title("Close Price with Moving Averages")
axes[0].grid(True, alpha=0.3)

# 4b. RSI
axes[1].plot(df.index, df["RSI_14"], color="purple", linewidth=1)
axes[1].axhline(70, color="red", linestyle="--", alpha=0.6, label="Overbought (70)")
axes[1].axhline(30, color="green", linestyle="--", alpha=0.6, label="Oversold (30)")
axes[1].set_ylabel("RSI")
axes[1].legend()
axes[1].set_title("Relative Strength Index (14-day)")
axes[1].grid(True, alpha=0.3)

# 4c. Volume
axes[2].bar(df.index, df["Volume"], color="steelblue", alpha=0.6, width=2)
axes[2].set_ylabel("Volume")
axes[2].set_title("Trading Volume")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("01_exploratory_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 01_exploratory_analysis.png")

# ──────────────────────────────────────────────
# 5.  BUILD THE PREDICTION MODEL
# ──────────────────────────────────────────────
features = ["SMA_10", "SMA_50", "RSI_14", "Daily_Return", "Volatility_10", "Volume_Change", "Momentum_5"]

X = df[features]
y = df["Target"]

# Train-test split (80/20, keeping temporal order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}\n")

# Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ──────────────────────────────────────────────
# 6.  EVALUATE MODEL ACCURACY
# ──────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.1f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"]))

# ──────────────────────────────────────────────
# 7.  CONFUSION MATRIX PLOT
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["DOWN", "UP"])
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix — Logistic Regression", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("02_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 02_confusion_matrix.png")

# ──────────────────────────────────────────────
# 8.  FEATURE IMPORTANCE
# ──────────────────────────────────────────────
coefs = pd.Series(model.coef_[0], index=features).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
coefs.plot(kind="barh", color=["#e74c3c" if v < 0 else "#2ecc71" for v in coefs], ax=ax)
ax.set_title("Feature Importance (Logistic Regression Coefficients)", fontsize=13, fontweight="bold")
ax.set_xlabel("Coefficient Value")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("03_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 03_feature_importance.png")

# ──────────────────────────────────────────────
# 9.  PREDICTION vs ACTUAL PLOT
# ──────────────────────────────────────────────
test_df = df.iloc[split_idx:].copy()
test_df["Predicted"] = y_pred

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_df.index, test_df["Close"], color="black", linewidth=1, label="Close Price")

up_correct = test_df[(test_df["Target"] == 1) & (test_df["Predicted"] == 1)]
down_correct = test_df[(test_df["Target"] == 0) & (test_df["Predicted"] == 0)]
wrong = test_df[test_df["Target"] != test_df["Predicted"]]

ax.scatter(up_correct.index, up_correct["Close"], color="green", marker="^", s=30, label="Correct UP", alpha=0.7)
ax.scatter(down_correct.index, down_correct["Close"], color="red", marker="v", s=30, label="Correct DOWN", alpha=0.7)
ax.scatter(wrong.index, wrong["Close"], color="gray", marker="x", s=20, label="Wrong", alpha=0.4)

ax.set_title("Predictions vs Actual — Test Period", fontsize=13, fontweight="bold")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("04_predictions_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Saved: 04_predictions_vs_actual.png")

print("\n" + "=" * 60)
print("ALL DONE! Check the generated PNG files for visualisations.")
print("=" * 60)
