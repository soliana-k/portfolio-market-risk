"""
Self-contained HTML report generation for the risk dashboard.

Produces a single HTML file (print-to-PDF friendly) summarising the portfolio,
risk metrics, model comparison and stress scenario results. No external
dependencies beyond pandas/plotly.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def _fmt_pct(v, dp: int = 2) -> str:
    try:
        return f"{float(v):.{dp}%}"
    except (TypeError, ValueError):
        return "-"


def _fmt_money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "-"


def _table(header: list[str], rows: list[list]) -> str:
    thead = "".join(f"<th>{h}</th>" for h in header)
    tbody = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def build_html_report(
    *,
    meta: dict,
    scheme: str,
    rebalance_freq: str,
    transaction_cost_bps: float,
    start_date,
    end_date,
    conf_str: str,
    target_weights: pd.Series,
    risk_metrics: dict,
    garch_res,
    gjr_res,
    ewma_res,
    cmp_results: dict,
    stress,
    correlation_df: pd.DataFrame,
    div_metrics: dict,
    multi_var_es: pd.DataFrame,
    attribution: pd.DataFrame,
    backtest_rows: list[dict],
) -> str:
    """Render a complete HTML report string."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # KPI summary
    kpi_rows = [
        ["Final Value", _fmt_money(meta["final_value"])],
        ["Total Return", _fmt_pct(meta["total_return"])],
        ["Rebalances", str(meta["n_rebalances"])],
        ["Trading Days", str(meta["n_days"])],
        ["Transaction Cost", f"{transaction_cost_bps} bps"],
    ]

    # Target weights
    tw = target_weights.sort_values(ascending=False)
    weight_rows = [[t, _fmt_pct(w)] for t, w in tw.items()]

    # Risk metrics
    risk_rows = [
        ["Historical VaR ($)", _fmt_money(risk_metrics["var_dollar"])],
        ["Historical VaR (%)", _fmt_pct(risk_metrics["var_pct"])],
        ["Expected Shortfall ($)", _fmt_money(risk_metrics["es_dollar"])],
        ["Expected Shortfall (%)", _fmt_pct(risk_metrics["es_pct"])],
    ]

    # Model comparison
    model_order = ["Historical", "EWMA", "GARCH", "GJR-GARCH"]
    cmp_rows = []
    for name in model_order:
        if name not in cmp_results:
            continue
        cr = cmp_results[name]
        cmp_rows.append([
            str(cr.rank),
            name,
            cr.backtest.basel_zone,
            f"{cr.backtest.n_exceptions}/{cr.backtest.n_observations}",
            _fmt_pct(cr.backtest.exception_ratio),
            _fmt_pct(cr.average_var),
            _fmt_pct(cr.expected_shortfall),
            f"{cr.backtest.kupiec_pof_pvalue:.3g}",
            cr.backtest.pass_fail,
        ])

    # Stress scenarios
    stress_rows = []
    for name in stress.ranking:
        sc = stress.scenarios[name]
        stress_rows.append([
            name,
            sc.scenario_type,
            _fmt_pct(sc.shock_return_pct),
            _fmt_pct(sc.stressed_var_pct),
            _fmt_pct(sc.stressed_es_pct),
            _fmt_money(sc.stressed_loss_value),
        ])

    # Multi-confidence VaR
    mc_rows = [
        [r["confidence"], _fmt_pct(r["var_pct"]),
         _fmt_money(r["var_dollar"]), _fmt_pct(r["es_pct"]),
         _fmt_money(r["es_dollar"])]
        for _, r in multi_var_es.iterrows()
    ]

    # Correlation table
    corr_rows = []
    for asset in correlation_df.columns:
        corr_rows.append([asset] + [f"{correlation_df.loc[asset, c]:.3f}" for c in correlation_df.columns])

    # Diversification
    div_rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in div_metrics.items()]

    # Attribution
    att_rows = [
        [r["asset"], _fmt_pct(r["total_return"]), _fmt_pct(r["weight"]),
         f"{r['contribution_pct']:.2f}%",
         (f"{r['contribution_share']:.2%}" if not pd.isna(r["contribution_share"]) else "-")]
        for _, r in attribution.iterrows()
    ]

    garch_json = {k: round(v, 6) for k, v in garch_res.params.items()}

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Portfolio Risk Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; color:#1a1a1a; margin:24px; }}
  h1 {{ font-size:22px; border-bottom:2px solid #d0d7de; padding-bottom:8px; }}
  h2 {{ font-size:16px; margin-top:28px; color:#0d6efd; }}
  .sub {{ color:#57606a; font-size:13px; }}
  table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:8px; }}
  th {{ text-align:left; background:#f6f8fa; padding:6px 8px; }}
  td {{ padding:5px 8px; border-bottom:1px solid #eaeef2; }}
  .grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:20px; }}
  .card {{ border:1px solid #d0d7de; border-radius:8px; padding:14px; }}
  .kpi {{ font-size:20px; font-weight:600; }}
</style></head><body>
<h1>Portfolio Market Risk Report</h1>
<div class="sub">Generated {now} &middot; Scheme: {scheme} &middot; Rebalance: {rebalance_freq} &middot;
Period: {start_date} to {end_date} &middot; Confidence: {conf_str}</div>

<h2>1. Portfolio Summary</h2>
{_table(["Metric", "Value"], kpi_rows)}

<h2>2. Target Weights</h2>
{_table(["Asset", "Weight"], weight_rows)}

<h2>3. Risk Metrics (Historical VaR)</h2>
{_table(["Metric", "Value"], risk_rows)}

<h2>4. VaR Model Comparison</h2>
{_table(["Rank", "Model", "Basel Zone", "Exceptions", "Ratio", "Avg VaR", "ES", "Kupiec p", "Result"], cmp_rows)}

<h2>5. Multi-Confidence VaR / ES</h2>
{_table(["Confidence", "VaR (%)", "VaR ($)", "ES (%)", "ES ($)"], mc_rows)}

<h2>6. Stress Scenarios</h2>
{_table(["Scenario", "Type", "Shock", "Stressed VaR", "Stressed ES", "Loss ($)"], stress_rows)}

<h2>7. Correlation Matrix</h2>
{_table(["(assets)"] + list(correlation_df.columns), corr_rows)}

<h2>8. Diversification</h2>
{_table(["Metric", "Value"], div_rows)}

<h2>9. Return Attribution</h2>
{_table(["Asset", "Total Return", "Weight", "Contribution (pp)", "Share"], att_rows)}

<h2>10. Volatility Model Parameters</h2>
<p>GARCH(1,1): {garch_json}</p>
<p>GJR-GARCH: omega={gjr_res.omega:.6f}, alpha={gjr_res.alpha:.6f},
beta={gjr_res.beta:.6f}, gamma (leverage)={gjr_res.leverage_parameter:.6f}</p>
<p>EWMA 1-day sigma forecast: {ewma_res.sigma_forecast:.2%}</p>
</body></html>"""

    return html


def export_csvs(**dataframes) -> dict[str, str]:
    """Return a dict of {filename: csv bytes} for the given named frames."""
    out = {}
    for name, df in dataframes.items():
        if isinstance(df, pd.DataFrame):
            out[f"{name}.csv"] = df.to_csv().encode("utf-8")
    return out
