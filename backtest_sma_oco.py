# backtest_sma_oco.py
# SMA(5/25/75) alignment entry + OCO(TP/SL) backtest with max concurrent positions
# Entry when SMA5 > SMA25 > SMA75 (no "newly" requirement)
# Enter next day's open, exit by OCO using daily OHLC (with gap handling)

import argparse
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# ---- watchlist ----
# watchlist_module.py must define: watchlist = {"1301.T": "極洋", ...}
from watchlist_module import watchlist


@dataclass
class Trade:
    code: str
    name: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str  # TP / SL / TIME
    tp: float
    sl: float
    hold_days: int
    ret_pct: float
    open_positions_at_entry: int


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def download_ohlc(code: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLC from yfinance.
    yfinance end is inclusive-ish depending; we pass end and accept what comes.
    """
    df = yf.download(code, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Normalize columns in case MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df


def add_sma(df: pd.DataFrame, windows=(5, 25, 75)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out


def choose_exit_price_on_gap(open_price: float, tp: float, sl: float, reason: str) -> float:
    """
    If open gaps beyond tp/sl, we assume fill at open.
    """
    if reason == "TP" and open_price >= tp:
        return open_price
    if reason == "SL" and open_price <= sl:
        return open_price
    return None  # means "not a gap fill"


def simulate_one_trade(
    code: str,
    name: str,
    df: pd.DataFrame,
    entry_idx: int,
    tp_mult: float,
    sl_mult: float,
    hold_max: int,
    priority: str,
    open_positions_at_entry: int,
) -> Optional[Trade]:
    """
    Entry at next day's Open after entry_idx (signal day).
    Then monitor from that day onward up to hold_max days.
    Exit by OCO using daily OHLC.
    """
    if entry_idx + 1 >= len(df):
        return None  # no next day to enter

    entry_day = df.index[entry_idx + 1]
    entry_open = float(df["Open"].iloc[entry_idx + 1])

    tp = entry_open * tp_mult
    sl = entry_open * sl_mult

    exit_reason = "TIME"
    exit_price = float(df["Close"].iloc[min(entry_idx + hold_max, len(df) - 1)])
    exit_day = df.index[min(entry_idx + hold_max, len(df) - 1)]
    hold_days = 0

    # We start checking from entry day itself (entry at open -> same day's high/low can hit)
    start_i = entry_idx + 1
    end_i = min(entry_idx + hold_max, len(df) - 1)

    for i in range(start_i, end_i + 1):
        day = df.index[i]
        o = float(df["Open"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        c = float(df["Close"].iloc[i])

        # Gap handling: if open already beyond levels
        if o >= tp:
            exit_reason = "TP"
            exit_price = o
            exit_day = day
            hold_days = i - start_i
            break
        if o <= sl:
            exit_reason = "SL"
            exit_price = o
            exit_day = day
            hold_days = i - start_i
            break

        hit_tp = h >= tp
        hit_sl = l <= sl

        if hit_tp and hit_sl:
            # both touched same day -> choose by priority
            if priority == "TP_first":
                exit_reason = "TP"
                exit_price = tp
            elif priority == "SL_first":
                exit_reason = "SL"
                exit_price = sl
            else:  # GapFair: decide by which is closer to open (crude but fair-ish)
                if abs(tp - o) < abs(o - sl):
                    exit_reason = "TP"
                    exit_price = tp
                else:
                    exit_reason = "SL"
                    exit_price = sl
            exit_day = day
            hold_days = i - start_i
            break

        if hit_tp:
            exit_reason = "TP"
            exit_price = tp
            exit_day = day
            hold_days = i - start_i
            break

        if hit_sl:
            exit_reason = "SL"
            exit_price = sl
            exit_day = day
            hold_days = i - start_i
            break

        # else continue until TIME exit

    ret_pct = (exit_price / entry_open) - 1.0

    return Trade(
        code=code,
        name=name,
        entry_date=str(entry_day.date()),
        entry_price=entry_open,
        exit_date=str(exit_day.date()),
        exit_price=exit_price,
        exit_reason=exit_reason,
        tp=tp,
        sl=sl,
        hold_days=hold_days,
        ret_pct=ret_pct,
        open_positions_at_entry=open_positions_at_entry,
    )


def build_summary(trades: pd.DataFrame, max_positions: int) -> pd.DataFrame:
    """
    Simple portfolio accounting:
    - Initial equity = 1.0
    - Each trade uses position_size = 1/max_positions
    - Cash sits idle if fewer than max_positions open.
    - Equity updates on exit dates (no mark-to-market).
    """
    if trades.empty:
        return pd.DataFrame([{
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret_pct": 0.0,
            "profit_factor": 0.0,
            "total_ret_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }])

    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    t = t.sort_values(["exit_date", "entry_date"]).reset_index(drop=True)

    pos_size = 1.0 / max_positions
    t["pnl"] = pos_size * t["ret_pct"]

    equity = 1.0
    equity_curve = []
    peak = 1.0
    max_dd = 0.0

    for _, row in t.iterrows():
        equity *= (1.0 + float(row["pnl"]))
        peak = max(peak, equity)
        dd = (equity / peak) - 1.0
        max_dd = min(max_dd, dd)
        equity_curve.append((row["exit_date"], equity))

    wins = t[t["ret_pct"] > 0.0]
    losses = t[t["ret_pct"] < 0.0]

    gross_profit = (pos_size * wins["ret_pct"]).sum()
    gross_loss = -(pos_size * losses["ret_pct"]).sum()

    pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    summary = {
        "trades": int(len(t)),
        "win_rate": float((t["ret_pct"] > 0).mean()),
        "avg_ret_pct": float(t["ret_pct"].mean()),
        "profit_factor": float(pf) if pf != float("inf") else 9999.0,
        "total_ret_pct": float(equity - 1.0),
        "max_drawdown_pct": float(max_dd),
    }
    return pd.DataFrame([summary])


def backtest(
    watch: str,
    start: str,
    end: str,
    tp: float,
    sl: float,
    hold_max: int,
    priority: str,
    limit: int,
    outdir: str,
    max_positions: int,
):
    safe_mkdir(outdir)

    if watch == "all":
        items = list(watchlist.items())
    else:
        # comma separated codes
        codes = [c.strip() for c in watch.split(",") if c.strip()]
        items = [(c, watchlist.get(c, c)) for c in codes]

    if limit and limit > 0:
        items = items[:limit]

    trades: List[Trade] = []

    # Track open positions by their exit_date (we know after sim, but for concurrency we need forward sim)
    # We'll do day-by-day scan per code and enforce max_positions by keeping a global "open until date" list.

    # Global list of (exit_date, code) for currently open positions at the moment of entering a new one.
    open_positions: List[Tuple[pd.Timestamp, str]] = []

    for idx, (code, name) in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {code} {name}")

        df = download_ohlc(code, start, end)
        if df.empty or len(df) < 80:
            print("  -> skip (no/short data)")
            continue

        df = add_sma(df)
        df = df.dropna().copy()
        if df.empty or len(df) < 2:
            print("  -> skip (no SMA data)")
            continue

        # Signal on day i (close known) => enter at day i+1 open.
        for i in range(len(df) - 1):
            curr = df.iloc[i]
            align = (curr["SMA5"] > curr["SMA25"] > curr["SMA75"])
            if not bool(align):
                continue

            signal_day = df.index[i]
            entry_day = df.index[i + 1]

            # Before entering, drop positions already closed by entry_day
            open_positions = [(ed, c) for (ed, c) in open_positions if ed >= entry_day]

            if len(open_positions) >= max_positions:
                # can't enter due to concurrency cap
                continue

            trade = simulate_one_trade(
                code=code,
                name=name,
                df=df,
                entry_idx=i,
                tp_mult=tp,
                sl_mult=sl,
                hold_max=hold_max,
                priority=priority,
                open_positions_at_entry=len(open_positions),
            )
            if trade is None:
                continue

            # Register this position as open until its exit date (inclusive)
            exit_day_ts = pd.to_datetime(trade.exit_date)
            open_positions.append((exit_day_ts, code))

            trades.append(trade)

    trades_df = pd.DataFrame([asdict(t) for t in trades])

    if trades_df.empty:
        print("No trades produced.")
        # still output empty files for convenience
        trades_path = os.path.join(outdir, "trades.csv")
        summary_path = os.path.join(outdir, "summary.csv")
        trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
        build_summary(trades_df, max_positions=max_positions).to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {trades_path}, {summary_path}")
        return

    # Add nice fields
    trades_df["ret_pct"] = trades_df["ret_pct"].astype(float)
    trades_df = trades_df.sort_values(["entry_date", "code"]).reset_index(drop=True)

    trades_path = os.path.join(outdir, "trades.csv")
    summary_path = os.path.join(outdir, "summary.csv")

    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary_df = build_summary(trades_df, max_positions=max_positions)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {trades_path}, {summary_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", required=True,
                   help="all もしくは '1301.T,1332.T' みたいにカンマ区切り")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (省略可)")
    p.add_argument("--tp", type=float, default=1.06, help="TP倍率（例: 1.06 = +6%）")
    p.add_argument("--sl", type=float, default=0.97, help="SL倍率（例: 0.97 = -3%）")
    p.add_argument("--hold-max", type=int, default=60, help="最大保有日数")
    p.add_argument("--priority", choices=["TP_first", "SL_first", "GapFair"], default="GapFair",
                   help="同日にTP/SL両方触れた場合の優先")
    p.add_argument("--limit", type=int, default=0, help="テスト銘柄数の上限（0なら無制限）")
    p.add_argument("--outdir", default="backtest_out", help="出力フォルダ")
    p.add_argument("--max-positions", type=int, default=3, help="同時保有の最大銘柄数")

    args = p.parse_args()

    end = args.end
    if end is None:
        # yfinance end is exclusive-ish; but ok. We'll set end to today+1
        end = (datetime.utcnow().date()).isoformat()

    backtest(
        watch=args.watch,
        start=args.start,
        end=end,
        tp=args.tp,
        sl=args.sl,
        hold_max=args.hold_max,
        priority=args.priority,
        limit=args.limit,
        outdir=args.outdir,
        max_positions=args.max_positions,
    )


if __name__ == "__main__":
    main()
