"""
Gate 1 v2 — measurement layer, not pass/fail.

Runs a signal DataFrame through a locked default execution wrapper at
unit sizing and reports six elements as STRONG/WEAK indicators against
malleable reference thresholds. The user (not this module) decides
whether the result is good enough to progress to Gate 2.

Key design choices (locked 2026-05-05, malleable on user command):
  * Default execution wrapper: 1.5x ATR stop, 2x ATR TP, EOD exit if
    neither hits, unit sizing (1 contract).
  * No path-survival as a binding criterion — Gate 1 is signal
    measurement; path-survival belongs in Gate 2.
  * No pass/fail. Six elements emit STRONG/WEAK relative to reference
    thresholds; user makes the call.

Reference thresholds in DEFAULT_GATE1_THRESHOLDS are forward-looking-
malleable: change them BEFORE running a hypothesis, never after seeing
the result.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.backtester import BacktestResult, run


# Locked default execution wrapper for Gate 1 v2.
# Override per-hypothesis via gate1_evaluate(..., wrapper_overrides=...)
# but document the override and the reason in the hypothesis pre-reg.
DEFAULT_GATE1_WRAPPER = {
    # Costs — pre-registered project default
    "commission_per_side": 2.5,        # $5 round-trip full NQ
    "spread_points": 0.25,
    "slippage_points": 0.0,

    # Execution
    "execution_delay_bars": 1,         # next-bar open
    "stop_atr_multiple": 1.5,
    "tp_atr_multiple": 2.0,
    "use_trailing_stop": False,
    "max_bars_in_trade": 0,            # let session_close / stop / TP handle exit
    "respect_session_close": True,     # EOD exit

    # Sizing — unit (Edge Mode style) so Gate 1 measures signal,
    # not compounding feedback
    "sizing_mode": "fixed_contract",
    "fixed_contracts": 1,

    # Halts off — Gate 1 is measurement, not deployment
    "daily_dd_limit": 0.0,
    "max_dd_limit": 0.0,
    "max_daily_risk": 0.0,
    "max_loss_per_trade": 0.0,
    "consec_loss_threshold": 999,

    # Concurrency relaxed so signals aren't silently dropped
    "max_concurrent_trades": 99,
    "min_bars_between_entries": 1,
}


# Reference thresholds for STRONG/WEAK classification.
# Malleable on user command (forward-looking only — change before run,
# never after seeing the result).
DEFAULT_GATE1_THRESHOLDS = {
    "pf_strong": 1.2,                  # profit factor net of costs
    "concentration_strong": 0.30,      # max single-day net P&L / total net P&L
    "profit_dd_strong": 1.5,           # net profit / max DD MTM
    "min_trades_strong": 30,           # sample size minimum
    "half_pf_floor": 1.0,              # both halves must exceed this
}


@dataclass
class Gate1Element:
    """Single Gate 1 measurement with its strong/weak verdict."""
    name: str
    value: float
    threshold: float
    direction: str             # "above" (strong if value >= threshold)
                               # "below" (strong if value <= threshold)
                               # "boolean" (strong if value == 1)
    verdict: str               # "STRONG" / "WEAK"
    note: str = ""


@dataclass
class Gate1Result:
    """Full Gate 1 measurement report."""
    elements: list = field(default_factory=list)
    n_trades: int = 0
    net_pnl_usd: float = 0.0
    backtest: BacktestResult | None = None
    diagnostics: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    wrapper_used: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "net_pnl_usd": self.net_pnl_usd,
            "elements": {e.name: e.verdict for e in self.elements},
        }


def _classify(value: float, threshold: float, direction: str) -> str:
    if direction == "above":
        return "STRONG" if value >= threshold else "WEAK"
    if direction == "below":
        return "STRONG" if value <= threshold else "WEAK"
    if direction == "boolean":
        return "STRONG" if value == 1 else "WEAK"
    raise ValueError(f"unknown direction {direction!r}")


def _profit_factor(pnls: np.ndarray) -> float:
    """Sum of wins / abs(sum of losses). Returns inf if no losses, 0 if no wins."""
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def _max_dd_mtm_usd(equity_mtm: pd.Series) -> float:
    """Peak-to-trough max drawdown in dollars from the MTM equity curve."""
    if len(equity_mtm) == 0:
        return 0.0
    running_max = equity_mtm.cummax()
    dd_usd = (equity_mtm - running_max)
    return float(-dd_usd.min())  # positive number


def _half_stability(pnls: np.ndarray, exit_dates: np.ndarray, half_pf_floor: float) -> tuple[bool, dict]:
    """Split trades by date midpoint. Both halves must be net positive
    AND have PF > half_pf_floor. Returns (is_strong, diagnostics)."""
    if len(pnls) < 2:
        return False, {"reason": "insufficient_trades"}

    unique_dates = np.array(sorted(set(exit_dates)))
    if len(unique_dates) < 2:
        return False, {"reason": "insufficient_date_span"}

    mid_date = unique_dates[len(unique_dates) // 2]
    first_mask = exit_dates < mid_date
    second_mask = ~first_mask

    pf1 = _profit_factor(pnls[first_mask])
    pf2 = _profit_factor(pnls[second_mask])
    net1 = float(pnls[first_mask].sum())
    net2 = float(pnls[second_mask].sum())

    diag = {
        "first_half_net_usd": round(net1, 2),
        "second_half_net_usd": round(net2, 2),
        "first_half_pf": round(pf1, 3) if not np.isinf(pf1) else float("inf"),
        "second_half_pf": round(pf2, 3) if not np.isinf(pf2) else float("inf"),
        "split_date": str(mid_date),
        "first_half_n": int(first_mask.sum()),
        "second_half_n": int(second_mask.sum()),
    }

    is_strong = (net1 > 0 and net2 > 0
                 and pf1 > half_pf_floor and pf2 > half_pf_floor)
    return is_strong, diag


def gate1_evaluate(
    signals: pd.DataFrame,
    wrapper_overrides: dict | None = None,
    thresholds: dict | None = None,
) -> Gate1Result:
    """Run Gate 1 v2 measurement on a signals DataFrame.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain open, high, low, close, signal, atr, session_close.
    wrapper_overrides : dict, optional
        Per-hypothesis overrides to DEFAULT_GATE1_WRAPPER. Use sparingly
        and document in pre-reg. Project-default behavior is the locked
        wrapper.
    thresholds : dict, optional
        Per-hypothesis overrides to DEFAULT_GATE1_THRESHOLDS. Forward-
        looking only — change before run, never after seeing result.

    Returns
    -------
    Gate1Result
    """
    wrapper = {**DEFAULT_GATE1_WRAPPER, **(wrapper_overrides or {})}
    th = {**DEFAULT_GATE1_THRESHOLDS, **(thresholds or {})}

    bt = run(signals, **wrapper)
    pnls = np.array([t.pnl for t in bt.trades]) if bt.trades else np.array([])
    exit_dates = np.array([t.exit_time.date() for t in bt.trades]) if bt.trades else np.array([])

    n_trades = len(pnls)
    net_pnl = float(pnls.sum()) if n_trades else 0.0

    # Element 1: profit factor net of costs
    pf = _profit_factor(pnls) if n_trades else 0.0
    el_pf = Gate1Element(
        name="profit_factor_net",
        value=round(pf, 3) if not np.isinf(pf) else float("inf"),
        threshold=th["pf_strong"],
        direction="above",
        verdict=_classify(pf, th["pf_strong"], "above"),
    )

    # Element 2: total net P&L positive
    el_total = Gate1Element(
        name="net_pnl_positive",
        value=1 if net_pnl > 0 else 0,
        threshold=0,
        direction="boolean",
        verdict="STRONG" if net_pnl > 0 else "WEAK",
        note=f"net ${net_pnl:,.2f}",
    )

    # Element 3: sample size
    el_n = Gate1Element(
        name="sample_size",
        value=n_trades,
        threshold=th["min_trades_strong"],
        direction="above",
        verdict=_classify(n_trades, th["min_trades_strong"], "above"),
    )

    # Element 4: single-day concentration
    if n_trades > 0:
        per_day = pd.Series(pnls).groupby(exit_dates).sum()
        max_day = float(per_day.max())
        if net_pnl > 0:
            concentration = max_day / net_pnl
        elif net_pnl < 0:
            concentration = float("inf")  # negative net P&L — concentration not meaningful
        else:
            concentration = 0.0
    else:
        concentration = 0.0
        max_day = 0.0

    el_conc = Gate1Element(
        name="single_day_concentration",
        value=round(concentration, 3) if not np.isinf(concentration) else float("inf"),
        threshold=th["concentration_strong"],
        direction="below",
        verdict=_classify(concentration, th["concentration_strong"], "below"),
        note=f"max single-day ${max_day:,.2f}",
    )

    # Element 5: half-stability
    is_stable, half_diag = _half_stability(pnls, exit_dates, th["half_pf_floor"]) \
        if n_trades > 0 else (False, {"reason": "no_trades"})
    el_half = Gate1Element(
        name="half_stability",
        value=1 if is_stable else 0,
        threshold=1,
        direction="boolean",
        verdict="STRONG" if is_stable else "WEAK",
        note=str(half_diag),
    )

    # Element 6: profit / max DD MTM
    max_dd_usd = _max_dd_mtm_usd(bt.equity_mtm) if n_trades else 0.0
    if max_dd_usd > 0:
        profit_dd = net_pnl / max_dd_usd
    elif net_pnl > 0:
        profit_dd = float("inf")
    else:
        profit_dd = 0.0

    el_dd = Gate1Element(
        name="profit_over_max_dd",
        value=round(profit_dd, 3) if not np.isinf(profit_dd) else float("inf"),
        threshold=th["profit_dd_strong"],
        direction="above",
        verdict=_classify(profit_dd, th["profit_dd_strong"], "above"),
        note=f"max DD ${max_dd_usd:,.2f}",
    )

    elements = [el_pf, el_total, el_n, el_conc, el_half, el_dd]

    diagnostics = {
        "max_single_day_pnl_usd": round(max_day, 2),
        "max_dd_mtm_usd": round(max_dd_usd, 2),
        "half_stability_detail": half_diag,
    }

    return Gate1Result(
        elements=elements,
        n_trades=n_trades,
        net_pnl_usd=round(net_pnl, 2),
        backtest=bt,
        diagnostics=diagnostics,
        thresholds=th,
        wrapper_used=wrapper,
    )


def print_gate1(result: Gate1Result) -> None:
    """Pretty-print a Gate 1 measurement report."""
    print("\n" + "=" * 72)
    print("  GATE 1 v2 - MEASUREMENT REPORT (no pass/fail)")
    print("=" * 72)
    print(f"  Trades: {result.n_trades:,}   Net P&L: ${result.net_pnl_usd:,.2f}")
    print("\n  Elements (STRONG/WEAK vs reference threshold):")
    print("  " + "-" * 68)
    for e in result.elements:
        if e.direction == "boolean":
            value_str = "yes" if e.value == 1 else "no"
            thresh_str = ""
        elif np.isinf(e.value):
            value_str = "inf"
            thresh_str = f"  (vs {e.direction} {e.threshold})"
        elif float(e.value).is_integer() and abs(e.value) < 1e9:
            value_str = f"{int(e.value):d}"
            thresh_str = f"  (vs {e.direction} {e.threshold})"
        else:
            value_str = f"{e.value:.3f}"
            thresh_str = f"  (vs {e.direction} {e.threshold})"
        print(f"    {e.name:30s} {value_str:>10s}{thresh_str:<22s}  [{e.verdict}]")
        if e.note and len(e.note) < 80:
            print(f"        - {e.note}")
    n_strong = sum(1 for e in result.elements if e.verdict == "STRONG")
    print("  " + "-" * 68)
    print(f"  Strong: {n_strong}/{len(result.elements)} elements.")
    print("  User decides whether the pattern warrants Gate 2.")
    print("=" * 72 + "\n")
