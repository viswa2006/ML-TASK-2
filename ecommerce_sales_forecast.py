"""
E-Commerce Sales Forecasting with Linear Models
=================================================
End-to-end example:
  1. Generate a realistic synthetic daily sales dataset
  2. Engineer time-series features (seasonality, lags, rolling stats)
  3. Train Linear Regression, Ridge, and Lasso models
  4. Evaluate with MAE / RMSE / MAPE / R^2 on a held-out time window
  5. Save charts + metrics + the dataset itself

Swap in your own data by replacing `generate_synthetic_data()` with a
`pd.read_csv(...)` call that produces a DataFrame with columns
['date', 'sales', 'promotion'] (promotion is optional).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "."


# ---------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ---------------------------------------------------------------------
def generate_synthetic_data(start_date="2023-01-01", periods=730):
    """Creates daily e-commerce sales with trend, weekly/yearly seasonality,
    a holiday-shopping surge, random promotions, and noise."""
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    n = len(dates)
    t = np.arange(n)

    # Long-term growth trend
    trend = 500 + 1.2 * t

    # Weekly seasonality: weekend + Friday boost
    dow = dates.dayofweek
    weekly = np.where(dow >= 5, 250, 0) + np.where(dow == 4, 100, 0)

    # Yearly seasonality (smooth sine wave, peaks in Q4)
    doy = dates.dayofyear.values
    yearly = 400 * np.sin(2 * np.pi * (doy - 80) / 365)

    # Holiday season lift (Nov-Dec)
    holiday_boost = np.where(dates.month.isin([11, 12]), 600, 0)

    # Black Friday / Cyber Monday style spikes
    bf_boost = np.zeros(n)
    for year in dates.year.unique():
        bf_start = pd.Timestamp(year=year, month=11, day=24)
        mask = (dates >= bf_start) & (dates <= bf_start + pd.Timedelta(days=3))
        bf_boost[mask] += 1500

    # Random promotions (~8% of days)
    promo_flag = np.random.binomial(1, 0.08, n)
    promo_effect = promo_flag * np.random.normal(400, 100, n)

    noise = np.random.normal(0, 150, n)

    sales = trend + weekly + yearly + holiday_boost + bf_boost + promo_effect + noise
    sales = np.clip(sales, 50, None)

    return pd.DataFrame({"date": dates, "sales": sales, "promotion": promo_flag})


# ---------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------
def create_features(df):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["time_index"] = np.arange(len(df))

    # Cyclical encodings so Jan/Dec and Sun/Mon are "close" to a linear model
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

    # Black Friday / Cyber Monday: a known calendar event a retailer plans
    # around, so (unlike organic demand noise) it's fair game as a feature.
    bf_flag = np.zeros(len(df), dtype=int)
    for year in df["date"].dt.year.unique():
        bf_start = pd.Timestamp(year=year, month=11, day=24)
        mask = (df["date"] >= bf_start) & (df["date"] <= bf_start + pd.Timedelta(days=3))
        bf_flag[mask.to_numpy()] = 1
    df["is_black_friday_week"] = bf_flag

    # Lag / rolling features (shifted so we never leak the current day)
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
    df["rolling_mean_30"] = df["sales"].shift(1).rolling(30).mean()

    return df.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------
# 3. TRAIN / EVALUATE
# ---------------------------------------------------------------------
def run_pipeline(test_size=60):
    raw = generate_synthetic_data()
    raw.to_csv(f"{OUTPUT_DIR}/synthetic_sales_data.csv", index=False)

    df = create_features(raw)

    feature_cols = [
        "time_index", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_weekend", "is_holiday_season", "is_black_friday_week", "promotion",
        "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_30",
    ]
    X, y = df[feature_cols], df["sales"]

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    dates_test = df["date"].iloc[-test_size:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1.0)": Ridge(alpha=1.0),
        "Lasso (alpha=1.0)": Lasso(alpha=1.0),
    }

    metrics_rows, preds_by_model, coefs_by_model = [], {}, {}

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        preds_by_model[name] = preds
        coefs_by_model[name] = model.coef_

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
        r2 = r2_score(y_test, preds)
        metrics_rows.append({"model": name, "MAE": mae, "RMSE": rmse,
                              "MAPE_%": mape, "R2": r2})

    metrics_df = pd.DataFrame(metrics_rows).set_index("model").round(3)
    metrics_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")
    print("\n=== Model comparison (60-day holdout) ===")
    print(metrics_df.to_string())

    best_name = metrics_df["RMSE"].idxmin()
    print(f"\nBest model by RMSE: {best_name}")

    plot_actual_vs_predicted(df, dates_test, y_test, preds_by_model, best_name)
    plot_residuals(dates_test, y_test, preds_by_model[best_name], best_name)
    plot_coefficients(feature_cols, coefs_by_model["Linear Regression"])

    return metrics_df, best_name


# ---------------------------------------------------------------------
# 4. PLOTS
# ---------------------------------------------------------------------
def plot_actual_vs_predicted(df, dates_test, y_test, preds_by_model, best_name):
    fig, ax = plt.subplots(figsize=(12, 5))
    history = df[df["date"] < dates_test.iloc[0]].tail(120)
    ax.plot(history["date"], history["sales"], color="#888888", label="Training history", linewidth=1)
    ax.plot(dates_test, y_test, color="#1f77b4", label="Actual", linewidth=2)
    ax.plot(dates_test, preds_by_model[best_name], color="#d62728",
             linestyle="--", label=f"Predicted ({best_name})", linewidth=2)
    ax.set_title("E-Commerce Daily Sales: Actual vs. Predicted (Holdout Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/forecast_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_residuals(dates_test, y_test, preds, best_name):
    residuals = y_test.values - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].scatter(dates_test, residuals, color="#2ca02c", s=20)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title(f"Residuals over Time ({best_name})")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual ($)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=15, color="#9467bd", edgecolor="white")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual ($)")

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/residual_analysis.png", dpi=150)
    plt.close(fig)


def plot_coefficients(feature_cols, coefs):
    order = np.argsort(np.abs(coefs))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in coefs[order]]
    ax.barh(np.array(feature_cols)[order], coefs[order], color=colors)
    ax.set_title("Linear Regression Coefficients (standardized features)")
    ax.set_xlabel("Coefficient value")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/feature_coefficients.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_pipeline()
