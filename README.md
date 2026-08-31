### Portfolio/Equity Market Risk 

## Phase 1: Data Collection



## 1. Project Overview

This repository contains the foundational data pipeline for the Equity Risk Platform. It handles automated data extraction from Yahoo Finance, computes financial return metrics, validates portfolio weights, and stores raw/processed artifacts securely under strict data integrity rules.

---

## 2. Project Directory Structure

```text
project/
├── src/
│   ├── data_collection/
│   │   ├── download_portfolio.py
│   │   ├── returns.py
│   │   └── build_portfolio.py
│   └── data_cleaning/
│       ├── __init__.py
│       └── cleaning.py     
├── data/
│   └── raw/                        
├── tests/                         
├── .github/
│   └── workflows/                  
├── requirements.txt                
└── README.md

```

---

## 3. Installation & Setup

### Prerequisites

Clone the repository and install the approved dependencies:

```bash
pip install -r requirements.txt

```

### Dependencies
see 
```text
requirements.txt
```

---

## 4. Testing & Verification

Run the automated test suite using `pytest` to verify technical correctness and business logic across all modules:

```bash
pytest tests/ -v

```

### Test Results & Verification Screenshot
**Testing Portfolio Download**

![alt text](test_download_portfolio_result.png)

**Testing Data Cleaning**
![test of data cleaning](test_cleaning_result.png)


## Phase 2: Data Cleaning

### Overview

Phase 2 cleans Phase 1 adjusted-close prices and builds the risk inputs used downstream:

- Clean price matrix  
- Clean log-return matrix (per asset)  
- Weighted portfolio return series  
- Rolling annualised portfolio volatility  

Steps: date alignment → missing-value handling → stale-price removal → validation → log returns → optional winsorisation → portfolio returns & rolling vol. Each step is logged with `loguru`.

### Module

```text
src/data_cleaning/
├── __init__.py
└── cleaning.py
```

Main entry point: `run_data_cleaning_pipeline(prices, weights)` → `CleanedDataResult`.

Optional diagnostics: `data_overview` / `print_overview` (shape, missingness, describe).

### Usage

```python
from src.data_cleaning import print_overview, run_data_cleaning_pipeline

print_overview(phase1["prices"])
result = run_data_cleaning_pipeline(phase1["prices"], phase1["weights"])

# result.portfolio_returns → VaR / GARCH / backtesting
# result.clean_returns, result.rolling_volatility, result.clean_prices
```

### Conventions

- Returns in **decimal** form (gains +, losses −)  
- Weights must sum to **1.0**  
- Volatility annualised with **√252**  
- No re-download — cleans Phase 1 outputs only  

### Tests

```bash
pytest tests/test_cleaning.py -v
```


## Phase 3: GARCH(1,1) Volatility & GARCH-Scaled Historical VaR

Fits a GARCH(1,1) model to the cleaned portfolio returns, estimates conditional
volatility, forecasts the 1-day-ahead volatility, and computes a GARCH-scaled
Historical Simulation VaR by scaling historical innovations with the forecast
volatility.

```text
src/garch_var/
├── __init__.py
└── garch.py
```

Main entry point: `fit_garch(returns, confidence_level=0.95, portfolio_value=None)`
→ `GarchResult` (conditional volatility, 1-day sigma forecast, standardised
residuals, GARCH-scaled returns, VaR / ES in % and $).

### Usage

```python
from src.garch_var import fit_garch

res = fit_garch(
    result.portfolio_returns,        # from Phase 2 cleaning pipeline
    confidence_level=0.95,
    portfolio_value=1_000_000,
)

res.sigma_forecast            # 1-day-ahead conditional volatility
res.conditional_volatility    # in-sample sigma_t series
res.standardized_residuals    # z_t = r_t / sigma_t (scaled innovations)
res.scaled_returns            # sigma_{t+1} * z_t simulated 1-day returns
res.var_pct, res.var_dollar  # GARCH-scaled Historical VaR
res.es_pct, res.es_dollar    # Expected Shortfall
```

### Method

1. Fit `arch.arch_model(returns, vol="Garch", p=1, q=1)` → conditional volatility `sigma_t`.
2. Standardise: `z_t = r_t / sigma_t` (i.i.d.-like innovations).
3. Forecast 1-day variance → `sigma_{t+1|t}`.
4. Scale historical innovations by the forecast: `r*_t = sigma_{t+1|t} * z_t`.
5. VaR / ES are the `alpha`-quantile (and tail mean) of the simulated `r*_t`.

Returns are internally rescaled by 100 for numerical stability (the `arch`
`DataScaleWarning` workaround) and parameters/volatility are rescaled back to
original units.

### Tests

```bash
pytest tests/test_garch.py -v
```

## Phase 7: GJR-GARCH Volatility Scaling

Extends Phase 3 with the **Glosten–Jagannathan–Runkle (GJR) GARCH** model, which
captures the **asymmetric / leverage effect**: negative returns raise future
volatility more than positive returns of the same magnitude (important for
equity portfolios).

```text
src/gjr_garch/
├── __init__.py
└── gjr.py
```

Main entry point: `fit_gjr_garch(returns, confidence_level=0.95, portfolio_value=None)`
→ `GjrGarchResult`.

### Why GJR-GARCH

- **Asymmetric volatility** — a negative shock impacts σ² more than a positive
  one of equal size.
- The variance recursion is

      σ²_t = ω + α·ε²_{t-1}
              + γ·ε²_{t-1}·I(ε_{t-1}<0)   ← leverage term
              + β·σ²_{t-1}

  so total sensitivity to bad news is **(α + γ)** vs **α** for good news.
- Useful for equity portfolios where crashes cluster.

### Implemented steps

1. Fit `GJR-GARCH(1,1,1)` via `arch.arch_model(vol="GARCH", o=1)` → conditional
   volatility `σ_t`.
2. Standardise returns `z_t = r_t / σ_t`.
3. Forecast 1-day-ahead volatility `σ_{t+1|t}`.
4. Scale historical innovations by the forecast: `r*_t = σ_{t+1|t}·z_t`.
5. Compute GJR-GARCH-scaled Historical Simulation **VaR / ES**.

### Output (`GjrGarchResult`)

- `conditional_volatility` — GJR-GARCH σ_t (in-sample)
- `leverage_parameter` (γ) — asymmetry / leverage coefficient
- `sigma_forecast` — 1-day-ahead conditional volatility
- `var_pct` / `var_dollar` (and `es_pct` / `es_dollar`) — GJR-GARCH-scaled VaR
- `garch_conditional_volatility` — plain GARCH(1,1) σ_t for direct comparison
- `leverage_stats` — empirical avg σ after negative vs positive return days

### Charts (in the app "Risk & VaR" tab)

- GJR-GARCH conditional volatility chart (+ 1-day forecast)
- GARCH vs GJR-GARCH volatility comparison
- Leverage-effect visualisation (next-day σ vs sign of today's return)
- Historical VaR vs GARCH-scaled vs GJR-GARCH-scaled VaR bar chart

### Usage

```python
from src.gjr_garch import fit_gjr_garch

res = fit_gjr_garch(
    result.portfolio_returns,
    confidence_level=0.95,
    portfolio_value=1_000_000,
)

res.leverage_parameter        # gamma (leverage effect)
res.sigma_forecast            # 1-day-ahead conditional volatility
res.conditional_volatility    # in-sample GJR sigma_t
res.var_pct, res.var_dollar  # GJR-GARCH-scaled Historical VaR
res.garch_conditional_volatility  # for GARCH vs GJR comparison
```

### Tests

```bash
pytest tests/test_gjr_garch.py -v
```

## Phase 8: VaR Backtesting

Checks whether the VaR models (Historical, GARCH-scaled, GJR-GARCH-scaled)
actually perform as advertised using a **rolling out-of-sample** backtest.

```text
src/var_backtesting/
├── __init__.py
└── backtest.py
```

Main entry points:
- `backtest_model(returns, model_func, model_name, confidence_level, estimation_window)`
  → `BacktestResult`
- `run_all_backtests(returns, confidence_level, estimation_window)` → dict of
  `BacktestResult` for every registered VaR model.

### Backtesting window

For each day `t` in the out-of-sample period (from `estimation_window` to the
end) the 1-day VaR is re-estimated using **only** the trailing
`estimation_window` observations (e.g. first 750 days for estimation, remaining
days for backtesting). An **exception** occurs when the realised loss exceeds
the VaR (`return_t < VaR_t`).

### Backtesting tests

- **Kupiec POF** (Proportion-of-Failures): likelihood-ratio test of correct
  *unconditional coverage* — does the exception rate match `1 - cl`?
- **Christoffersen Independence**: LR test that exceptions are **not clustered**
  (no autocorrelation in the hit sequence).
- **Christoffersen Conditional Coverage**: LR test (df = 2) combining correct
  coverage **and** independence.
- **Basel Traffic Light**: one-sided zone (Green / Yellow / Red) from the
  upper-tail probability `P(X ≥ x)` of `Binomial(N, 1-cl)`. Calibrated to
  reproduce the canonical Basel boundaries for a 99% VaR over 250 observations:
  **Green 0–4, Yellow 5–9, Red ≥10** exceptions.

### Outputs (`BacktestResult`)

- `n_exceptions`, `expected_exceptions`, `exception_ratio`
- `kupiec_pof_pvalue`, `christoffersen_independence_pvalue`,
  `christoffersen_cc_pvalue`
- `basel_zone` (Green / Yellow / Red)
- `pass_fail` (Pass / Fail — Green zone and no independence rejection)

### Charts / UI (app "🚦 Backtest" tab)

- Per-model summary table (exceptions, expected, ratios, p-values, zone, result)
- VaR forecast vs realised returns timeline with exceptions highlighted
- Exceptions-vs-expected bar chart coloured by Basel zone

### Usage

```python
from src.var_backtesting import run_all_backtests

bt = run_all_backtests(
    portfolio_returns,
    confidence_level=0.95,
    estimation_window=750,
)
for name, res in bt.items():
    print(name, res.basel_zone, res.pass_fail, res.n_exceptions)
```

### Tests

```bash
pytest tests/test_var_backtesting.py -v
```

## Phase 9: VaR Model Comparison

Compares four VaR methodologies on the **same** rolling out-of-sample backtest:

1. **Historical Simulation VaR**
2. **EWMA-scaled Historical VaR** (RiskMetrics, λ = 0.94)
3. **GARCH-scaled Historical VaR**
4. **GJR-GARCH-scaled Historical VaR**

### EWMA (RiskMetrics) model — `src/ewma_var/ewma.py`

Exponentially Weighted Moving-Average volatility:

    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

1-day variance forecast = `lambda * sigma_t^2 + (1 - lambda) * r_t^2`.
Historical innovations `z_t = r_t / sigma_t` are scaled by the forecast to
build the simulated return distribution, and VaR / ES are read from its
quantiles (same recipe as the GARCH-scaled models).

### Comparison metrics (`src/model_comparison/compare.py`)

For every model, over the backtest period:

- **Number of exceptions** and **exception ratio**
- **Average VaR** (mean of the VaR series) and **VaR volatility** (std)
- **Worst daily loss** (minimum realised return)
- **Expected Shortfall** (empirical mean return on exception days)
- **Backtesting p-values** (Kupiec POF, Christoffersen independence & CC)
- **Traffic-light zone** (Green / Yellow / Red)

A **model ranking** orders models by Basel zone (Green best), then by the
absolute deviation of the realised exception ratio from the target, then by the
Kupiec p-value.

### Charts / UI (app "📊 Model Comparison" tab)

- VaR model comparison line chart (all four VaR series over time)
- Actual P&L vs VaR chart (focus model)
- Exceptions overlay chart (returns with exception days highlighted)
- Backtesting summary bar chart (expected vs observed exceptions per model)
- Model ranking table

### Usage

```python
from src.model_comparison import compare_models

cmp = compare_models(portfolio_returns, confidence_level=0.95, estimation_window=750)
for name, cr in cmp.items():
    print(name, cr.rank, cr.backtest.basel_zone, cr.average_var, cr.expected_shortfall)
```

### Tests

```bash
pytest tests/test_ewma.py tests/test_model_comparison.py -v
```

## Phase 10: Stress Testing

Applies a battery of stress scenarios to the current portfolio to estimate
losses, stressed VaR and stressed Expected Shortfall beyond the in-sample
distribution. Two scenario families:

- **Historical scenarios** (`HISTORICAL_SCENARIOS`): 2008 Financial Crisis,
  COVID Crash 2020, Inflation Shock 2022, Banking Stress 2023, Tech Selloff.
  When the crisis window falls inside the portfolio's return history the
  scenario uses the **actual realised loss** over that window; otherwise a
  representative equity drawdown is used.
- **Hypothetical scenarios** (`HYPOTHETICAL_SCENARIOS`): Equity −5% / −10% /
  −20% (instant proportional shock), Volatility ×2 (VaR scaled by 2),
  Correlation Spike (VaR scaled by 1.5), and a Sector Shock that hits the
  single largest position with a −15% move, with per-asset loss contributions
  recorded.

### Output (`src/stress_testing/stress.py`)

`run_stress_tests(portfolio_value, weights, portfolio_returns, asset_returns,
baseline_var_pct, baseline_es_pct, confidence_level=0.95)` returns a
`StressTestResults` object:

- `scenarios` — `dict[name -> ScenarioResult]` with `scenario_type`,
  `shock_return_pct`, `stressed_var_pct`, `stressed_es_pct`,
  `stressed_loss_value`, `source` (actual vs representative), `worst_case`,
  and (for sector shock) `sector_contrib` (per-asset loss Series).
- `ranking` — scenario names ordered by `stressed_loss_value` (worst first).
- `worst_case_name` — the highest-loss scenario.

Stressed VaR / ES are computed relative to the supplied baseline (GJR-GARCH)
VaR / ES, with equity and sector shocks adding directly to the loss.

### Charts / UI (app "🚨 Stress" tab)

- Stress scenario results table + scenario loss bar chart
- Stress VaR comparison (stressed vs baseline VaR per scenario)
- Worst loss ranking bar chart
- Sector shock impact (loss contribution by asset)
- Drawdown + daily return distribution
- Rolling Historical VaR vs actual returns
- VaR vs Actual Loss (all four models over the backtest period)
- VaR vs Expected Shortfall (per model, full sample)
- Volatility regime comparison (rolling 21d, EWMA, GARCH, GJR-GARCH)
- Basel traffic light + exceptions by model + expected vs actual exceptions

### Usage

```python
from src.stress_testing import run_stress_tests

stress = run_stress_tests(
    portfolio_value=1_000_000,
    weights=result.target_weights,
    portfolio_returns=result.portfolio_returns,
    asset_returns=clean_prices.pct_change().dropna(),
    baseline_var_pct=gjr_res.var_pct,
    baseline_es_pct=gjr_res.es_pct,
    confidence_level=0.95,
)
print(stress.worst_case_name, stress.ranking)
```

### Tests

```bash
pytest tests/test_stress.py -v
```

## Phase 4:

```bash
streamlit run app.py
```