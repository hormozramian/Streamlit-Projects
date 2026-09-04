"""
Two-Sleeve Portfolio Construction Explorer (Stocks + Bonds)
--------------------------------------------------------------
An interactive Streamlit app for building a combined stock and bond
portfolio under three governing controls (risk appetite, liquidity
requirement, sector diversification), compared across five portfolio
construction approaches: Maximum Sharpe Ratio, Kelly / growth-optimal,
Mean-Variance (target return), Minimum Variance, and Risk Parity.

All assets, expected returns, and the covariance matrix are synthetic,
built from a small factor model purely to illustrate how each method
behaves, not a real investable universe.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Stock & Bond Portfolio Explorer", layout="wide")

RF = 0.02
STOCK_COLOR = "#2E86AB"
BOND_COLOR = "#C73E1D"

SECTOR_COLORS = {
    "Technology": "#2E86AB", "Financials": "#6BAED6", "Healthcare": "#74C476",
    "Energy": "#FD8D3C", "Consumer Staples": "#9E9AC8", "Industrials": "#969696",
    "Real Estate": "#E7298A", "Government": "#C73E1D", "Corporate IG": "#F16913",
    "Corporate HY": "#8C510A", "Municipal": "#B15928",
}

METHOD_OPTIONS = ["Maximum Sharpe Ratio", "Kelly / Growth-Optimal", "Mean-Variance (target return)",
                   "Minimum Variance", "Risk Parity"]

RISK_APPETITE_MAP = {"Conservative": 0.35, "Moderate": 0.65, "Aggressive": 1.00}
LIQUIDITY_MAP = {"Any liquidity": 1, "Medium and above": 4, "High liquidity only": 8}
SECTOR_CAP_MAP = {"Concentrated": 0.60, "Balanced": 0.35, "Highly diversified": 0.20}


# ---------------------------------------------------------------------------
# Universe and covariance construction (cached; identical every run)
# ---------------------------------------------------------------------------

@st.cache_data
def build_universe():
    assets = pd.DataFrame([
        ("Tech_A",            "Stock", "Technology",        0.12,  0.28, 9,  1.30, 0.00, 0.00),
        ("Tech_B",             "Stock", "Technology",        0.13,  0.32, 6,  1.45, 0.00, 0.00),
        ("Financials_A",       "Stock", "Financials",        0.09,  0.24, 9,  1.10, 0.20, 0.10),
        ("Healthcare_A",       "Stock", "Healthcare",        0.08,  0.18, 8,  0.70, 0.00, 0.00),
        ("Energy_A",           "Stock", "Energy",            0.10,  0.30, 7,  0.95, 0.00, 0.15),
        ("ConsumerStaples_A",  "Stock", "Consumer Staples",  0.06,  0.14, 9,  0.55, 0.00, 0.00),
        ("Industrials_A",      "Stock", "Industrials",       0.09,  0.22, 7,  1.05, 0.05, 0.05),
        ("SmallCapREIT_A",     "Stock", "Real Estate",       0.11,  0.33, 2,  0.90, 0.35, 0.20),
        ("Govt_Short",         "Bond",  "Government",        0.025, 0.02, 10, 0.00, 0.15, 0.00),
        ("Govt_Long",          "Bond",  "Government",        0.045, 0.09, 9,  0.00, 0.75, 0.00),
        ("Corp_IG",            "Bond",  "Corporate IG",      0.050, 0.07, 7,  0.05, 0.45, 0.35),
        ("Corp_HY",            "Bond",  "Corporate HY",      0.075, 0.13, 5,  0.20, 0.30, 0.65),
        ("Muni_A",             "Bond",  "Municipal",         0.040, 0.06, 3,  0.00, 0.40, 0.20),
    ], columns=["name", "asset_class", "sector", "exp_return", "volatility", "liquidity",
                "mkt_beta", "rate_beta", "credit_beta"])

    factor_vol = np.array([0.16, 0.09, 0.08])
    factor_corr = np.array([[1.00, -0.15, 0.30], [-0.15, 1.00, 0.10], [0.30, 0.10, 1.00]])
    factor_cov = np.outer(factor_vol, factor_vol) * factor_corr

    loadings = assets[["mkt_beta", "rate_beta", "credit_beta"]].values
    systematic_cov = loadings @ factor_cov @ loadings.T
    target_var = assets["volatility"].values ** 2
    idio_var = np.maximum(target_var - np.diag(systematic_cov), 0.0005)
    cov = systematic_cov + np.diag(idio_var)

    mu = assets["exp_return"].values
    return assets, mu, cov


# ---------------------------------------------------------------------------
# Constraint builder and the five optimizers
# ---------------------------------------------------------------------------

def build_constraints(assets, max_stock_pct, min_liquidity, sector_cap):
    n = len(assets)
    bounds = [(0.0, 0.0) if row["liquidity"] < min_liquidity else (0.0, 1.0)
              for _, row in assets.iterrows()]

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    stock_idx = assets.index[assets["asset_class"] == "Stock"].to_numpy()
    cons.append({"type": "ineq", "fun": lambda w, idx=stock_idx: max_stock_pct - w[idx].sum()})
    for sector in assets["sector"].unique():
        idx = assets.index[assets["sector"] == sector].to_numpy()
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: sector_cap - w[idx].sum()})
    return bounds, cons


def solve_max_sharpe(mu, cov, bounds, cons):
    n = len(mu)
    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - RF) / vol if vol > 1e-8 else 1e6
    res = minimize(neg_sharpe, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def solve_kelly(mu, cov, bounds, cons):
    n = len(mu)
    def neg_growth(w):
        return -(w @ mu - 0.5 * w @ cov @ w)
    res = minimize(neg_growth, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def solve_mean_variance(mu, cov, bounds, cons, target_return):
    n = len(mu)
    cons2 = cons + [{"type": "eq", "fun": lambda w: w @ mu - target_return}]
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                    constraints=cons2, options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def solve_min_variance(cov, bounds, cons):
    n = cov.shape[0]
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def solve_risk_parity(cov, bounds, cons):
    n = cov.shape[0]
    def rp_objective(w):
        port_var = w @ cov @ w
        risk_contrib = w * (cov @ w)
        return np.sum((risk_contrib - port_var / n) ** 2)
    res = minimize(rp_objective, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                    constraints=cons, options={"maxiter": 1000, "ftol": 1e-14})
    return res.x


def solve_portfolio(method, mu, cov, bounds, cons, target_return):
    if method == "Maximum Sharpe Ratio":
        return solve_max_sharpe(mu, cov, bounds, cons)
    if method == "Kelly / Growth-Optimal":
        return solve_kelly(mu, cov, bounds, cons)
    if method == "Mean-Variance (target return)":
        return solve_mean_variance(mu, cov, bounds, cons, target_return)
    if method == "Minimum Variance":
        return solve_min_variance(cov, bounds, cons)
    if method == "Risk Parity":
        return solve_risk_parity(cov, bounds, cons)
    raise ValueError(method)


def risk_contributions(w, cov):
    port_var = w @ cov @ w
    if port_var < 1e-12:
        return np.zeros_like(w)
    return (w * (cov @ w)) / port_var


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Portfolio Governance Controls")

risk_appetite = st.sidebar.selectbox("Risk Appetite", list(RISK_APPETITE_MAP.keys()), index=1,
                                      help="Caps the maximum combined weight allowed in the stock sleeve.")
liquidity_pref = st.sidebar.selectbox("Liquidity Requirement", list(LIQUIDITY_MAP.keys()), index=1,
                                       help="Assets below the required liquidity score are excluded entirely.")
sector_div = st.sidebar.selectbox("Sector Diversification", list(SECTOR_CAP_MAP.keys()), index=1,
                                   help="Caps the maximum combined weight in any single sector.")

st.sidebar.markdown("---")
method = st.sidebar.selectbox("Portfolio Construction Approach", METHOD_OPTIONS, index=0)

target_return = None
if method == "Mean-Variance (target return)":
    target_return = st.sidebar.slider("Target annual return", min_value=0.03, max_value=0.11,
                                       value=0.06, step=0.005, format="%.3f")

# ---------------------------------------------------------------------------
# Solve and display
# ---------------------------------------------------------------------------

st.title("Stock and Bond Portfolio Construction Explorer")
st.caption(
    "A two-sleeve universe of 8 stocks and 5 bonds. Risk appetite, liquidity, and sector "
    "diversification set the constraints; the selected method solves the portfolio within them. "
    "All data is synthetic."
)

assets, mu, cov = build_universe()
max_stock_pct = RISK_APPETITE_MAP[risk_appetite]
min_liquidity = LIQUIDITY_MAP[liquidity_pref]
sector_cap = SECTOR_CAP_MAP[sector_div]

bounds, cons = build_constraints(assets, max_stock_pct, min_liquidity, sector_cap)

feasible_target = None
if target_return is not None:
    feasible_target = float(np.clip(target_return, mu.min() + 0.002, mu.max() - 0.002))

try:
    w = solve_portfolio(method, mu, cov, bounds, cons, feasible_target)
    solve_ok = True
except Exception as exc:
    solve_ok = False
    st.error(f"The optimizer could not find a feasible portfolio under these constraints: {exc}")

if solve_ok:
    port_return = w @ mu
    port_vol = np.sqrt(w @ cov @ w)
    port_sharpe = (port_return - RF) / port_vol if port_vol > 1e-8 else np.nan
    stock_pct = w[assets["asset_class"] == "Stock"].sum()
    bond_pct = w[assets["asset_class"] == "Bond"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Return", f"{port_return:.2%}")
    col2.metric("Volatility", f"{port_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{port_sharpe:.2f}")
    col4.metric("Stock / Bond Split", f"{stock_pct:.0%} / {bond_pct:.0%}")

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Portfolio Weights")
        weights_df = assets[["name", "asset_class", "sector"]].copy()
        weights_df["weight"] = w
        weights_df = weights_df[weights_df["weight"] > 1e-4].sort_values("weight", ascending=True)

        fig_w, ax_w = plt.subplots(figsize=(6.5, 5.5))
        bar_colors = [STOCK_COLOR if ac == "Stock" else BOND_COLOR for ac in weights_df["asset_class"]]
        ax_w.barh(weights_df["name"], weights_df["weight"], color=bar_colors)
        ax_w.set_xlabel("Portfolio weight")
        ax_w.set_title(f"{method}\n(excluded / zero-weight assets not shown)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=STOCK_COLOR), plt.Rectangle((0, 0), 1, 1, color=BOND_COLOR)]
        ax_w.legend(handles, ["Stock", "Bond"], loc="lower right")
        st.pyplot(fig_w)

    with right:
        st.subheader("Risk Contribution by Asset")
        rc = risk_contributions(w, cov)
        rc_df = assets[["name", "sector"]].copy()
        rc_df["risk_contribution"] = rc
        rc_df = rc_df[rc_df["risk_contribution"] > 1e-4].sort_values("risk_contribution", ascending=True)

        fig_rc, ax_rc = plt.subplots(figsize=(6.5, 5.5))
        rc_colors = [SECTOR_COLORS.get(s, "#888888") for s in rc_df["sector"]]
        ax_rc.barh(rc_df["name"], rc_df["risk_contribution"], color=rc_colors)
        ax_rc.set_xlabel("Share of total portfolio risk")
        ax_rc.set_title("Where the Portfolio's Risk Actually Comes From")
        st.pyplot(fig_rc)

    st.markdown("---")
    st.subheader("Risk / Return Map")
    st.caption(
        "Every individual asset (small dots), plus the portfolio chosen by each of the five methods "
        "(large markers) under the SAME governance controls set in the sidebar. Methods that tolerate "
        "more risk (Kelly) sit further right; the minimum-variance portfolio sits furthest left."
    )

    method_points = {}
    for m in METHOD_OPTIONS:
        try:
            tr = feasible_target if m == "Mean-Variance (target return)" else None
            if m == "Mean-Variance (target return)" and tr is None:
                tr = 0.06
            w_m = solve_portfolio(m, mu, cov, bounds, cons, tr)
            method_points[m] = (np.sqrt(w_m @ cov @ w_m), w_m @ mu)
        except Exception:
            pass

    fig_map, ax_map = plt.subplots(figsize=(10, 6))
    asset_colors = [STOCK_COLOR if ac == "Stock" else BOND_COLOR for ac in assets["asset_class"]]
    ax_map.scatter(assets["volatility"], assets["exp_return"], c=asset_colors, s=60, alpha=0.55,
                    edgecolor="white", zorder=3)
    for _, row in assets.iterrows():
        ax_map.annotate(row["name"], (row["volatility"], row["exp_return"]), fontsize=7,
                         xytext=(4, 3), textcoords="offset points", alpha=0.7)

    marker_shapes = {"Maximum Sharpe Ratio": "*", "Kelly / Growth-Optimal": "D",
                      "Mean-Variance (target return)": "s", "Minimum Variance": "P", "Risk Parity": "o"}
    marker_colors = {"Maximum Sharpe Ratio": "#7209B7", "Kelly / Growth-Optimal": "#F18F01",
                      "Mean-Variance (target return)": "#2E86AB", "Minimum Variance": "#6A994E",
                      "Risk Parity": "#C73E1D"}
    for m, (v, r) in method_points.items():
        is_selected = (m == method)
        ax_map.scatter([v], [r], marker=marker_shapes[m], s=380 if is_selected else 220,
                        color=marker_colors[m], edgecolor="black" if is_selected else "white",
                        linewidth=2 if is_selected else 1, zorder=5, label=m + ("  (selected)" if is_selected else ""))

    ax_map.set_xlabel("Annualized volatility"); ax_map.set_ylabel("Expected annual return")
    ax_map.set_title("Assets and the Five Portfolio Construction Methods")
    ax_map.legend(loc="upper left", fontsize=8)
    st.pyplot(fig_map)

    if liquidity_pref != "Any liquidity":
        excluded_names = assets.loc[[b == (0.0, 0.0) for b in bounds], "name"].tolist()
        if excluded_names:
            st.caption(f"Excluded by the '{liquidity_pref}' liquidity requirement: {', '.join(excluded_names)}")

with st.expander("How this demo works"):
    st.markdown(
        """
        Thirteen synthetic assets, 8 stocks across 7 sectors and 5 bonds across 4 categories, are
        built from a 3-factor risk model (a market/equity factor, a rate/duration factor, and a
        credit factor), which guarantees a valid, positive semi-definite covariance matrix. The three
        sidebar controls above the method selector each map onto a genuine constraint in the
        underlying optimization: Risk Appetite caps the combined stock-sleeve weight, Liquidity
        Requirement removes any asset below the selected liquidity score from the eligible universe
        entirely (its weight is fixed at exactly zero, not merely discouraged), and Sector
        Diversification caps how much weight any single sector can carry.

        Within those shared constraints, the five methods solve genuinely different objectives:
        Maximum Sharpe Ratio maximizes risk-adjusted return; Kelly / Growth-Optimal maximizes expected
        long-run compound growth (mean minus half-variance, the standard continuous-time
        approximation), which characteristically takes on more risk than Sharpe-maximizing allocations
        given the chance; Mean-Variance minimizes risk for a chosen target return; Minimum Variance
        ignores expected return entirely and minimizes risk outright; and Risk Parity looks for the
        allocation where every holding contributes as close to an equal share of total portfolio risk
        as the other active constraints allow, note that with sector caps and a stock/bond limit also
        binding at the same time, true risk parity's equal-contribution goal is only ever approximately
        achieved, not exactly, which is the correct, honest behavior for a constrained risk parity
        solve rather than a modeling error.
        """
    )
