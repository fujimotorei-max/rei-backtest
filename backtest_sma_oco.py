import os
import json
import argparse
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

def normalize_ohlc(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    yfinance が MultiIndex で返す/同名列が複数になるケースを吸収して、
    Open/High/Low/Close を1列ずつに正規化する
    """
    if df is None or df.empty:
        return df

    # MultiIndex columns 対応
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(code, axis=1, level=1, drop_level=True)
        except Exception:
            df.columns = [c[0] for c in df.columns]

    # Closeなどが DataFrame になってたら先頭列を使う
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df


# -----------------------------
# Watchlist loader
# -----------------------------
def load_watchlist(path: str) -> Dict[str, str]:
    """
    Accept:
      - .py file exporting `watchlist` dict (e.g., {"1301.T": "極洋", ...})
      - .json file containing a dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"--watch '{path}' が見つからないよ。大文字小文字も含めて、リポジトリのファイル名と完全一致させてね。"
        )

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            wl = json.load(f)
        if not isinstance(wl, dict):
            raise ValueError("watchlist json は dict 形式じゃないとだめ。")
        return wl

    if path.endswith(".py"):
        ns = {}
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, path, "exec"), ns, ns)
        wl = ns.get("watchlist")
        if not isinstance(wl, dict):
            raise ValueError(f"{path} に watchlist (dict) が見つからないよ。")
        return wl

    raise ValueError("--watch は .py か .json を指定してね。")


# -----------------------------
# Strategy
# -----------------------------
@dataclass
class Trade:
    code: str
    name: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    exit_reason: str  # TP / SL / TIME / EOD
    hold_days: int
    ret: float


def add_sma(df: pd.DataFrame, windows=(5, 25, 75)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out


def entry_signal_align_only(row: pd.Series) -> bool:
    """5 > 25 > 75 ならエントリー（新規判定いらない）"""
    return (row["SMA5"] > row["SMA25"]) and (row["SMA25"] > row["SMA75"])


def simulate_oco_exit(
    df: pd.DataFrame,
    entry_i: int,
    tp: float,
    sl: float,
    hold_max: int,
    priority: str,
) -> Tuple[int, float, str]:
    """
    Entry at next day's Open (to avoid look-ahead).
    Then simulate OCO using daily High/Low.
    priority:
      - SL_first: if both hit same day -> SL
      - TP_first: -> TP
      - GapFair : if Open gaps beyond level, use Open price; if both beyond, choose closer (rough)
    Returns: (exit_index, exit_price, reason)
    """
    # entry happens at entry_i (already next day)
    entry_price = float(df.iloc[entry_i]["Open"])
    tp_price = entry_price * (1.0 + tp)
    sl_price = entry_price * (1.0 - sl)

    last_i = min(entry_i + hold_max - 1, len(df) - 1)

    for i in range(entry_i, last_i + 1):
        o = float(df.iloc[i]["Open"])
        h = float(df.iloc[i]["High"])
        l = float(df.iloc[i]["Low"])

        # Gap handling
        if priority == "GapFair":
            # If open already beyond TP/SL, assume executed at open.
            if o >= tp_price and o <= sl_price:
                # practically impossible unless tp_price <= sl_price, but keep safe
                return i, o, "EOD"
            if o >= tp_price:
                return i, o, "TP"
            if o <= sl_price:
                return i, o, "SL"

        hit_tp = h >= tp_price
        hit_sl = l <= sl_price

        if hit_tp and hit_sl:
            if priority == "TP_first":
                return i, tp_price, "TP"
            if priority == "SL_first":
                return i, sl_price, "SL"
            # GapFair: decide by which is closer to Open (rough intraday order proxy)
            if abs(o - tp_price) < abs(o - sl_price):
                return i, tp_price, "TP"
            else:
                return i, sl_price, "SL"
        elif hit_tp:
            return i, tp_price, "TP"
        elif hit_sl:
            return i, sl_price, "SL"

    # time exit at last day close
    return last_i, float(df.iloc[last_i]["Close"]), "TIME"


def backtest_one(
    code: str,
    name: str,
    start: str,
    end: Optional[str],
    tp: float,
    sl: float,
    hold_max: int,
    priority: str,
) -> List[Trade]:
    raw = yf.download(code, start=start, end=end, interval="1d", progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return []

    df = raw.copy()
    df = add_sma(df)
    df = df.dropna().reset_index()

    if df.empty or len(df) < 2:
        return []

    trades: List[Trade] = []
    in_pos = False
    i = 0

    while i < len(df) - 1:
        row = df.iloc[i]
        if not in_pos:
            if entry_signal_align_only(row):
                # Enter next day open
                entry_i = i + 1
                exit_i, exit_price, reason = simulate_oco_exit(
                    df=df,
                    entry_i=entry_i,
                    tp=tp,
                    sl=sl,
                    hold_max=hold_max,
                    priority=priority,
                )
                entry_price = float(df.iloc[entry_i]["Open"])
                entry_date = str(df.iloc[entry_i]["Date"].date())
                exit_date = str(df.iloc[exit_i]["Date"].date())
                hold_days = int(exit_i - entry_i + 1)
                ret = (exit_price / entry_price) - 1.0

                trades.append(
                    Trade(
                        code=code,
                        name=name,
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=exit_date,
                        exit_price=exit_price,
                        exit_reason=reason,
                        hold_days=hold_days,
                        ret=ret,
                    )
                )
                # after exit, continue scanning from exit day
                i = exit_i
            else:
                i += 1
        else:
            i += 1

    return trades


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "code","name","entry_date","entry_price","exit_date","exit_price",
                "exit_reason","hold_days","ret"
            ]
        )
    return pd.DataFrame([t.__dict__ for t in trades])


def summary_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret": 0.0,
            "median_ret": 0.0,
            "total_ret_simple_sum": 0.0,
            "avg_hold_days": 0.0,
            "tp_count": 0,
            "sl_count": 0,
            "time_count": 0,
        }])

    win_rate = float((df["ret"] > 0).mean())
    return pd.DataFrame([{
        "trades": int(len(df)),
        "win_rate": win_rate,
        "avg_ret": float(df["ret"].mean()),
        "median_ret": float(df["ret"].median()),
        "total_ret_simple_sum": float(df["ret"].sum()),
        "avg_hold_days": float(df["hold_days"].mean()),
        "tp_count": int((df["exit_reason"] == "TP").sum()),
        "sl_count": int((df["exit_reason"] == "SL").sum()),
        "time_count": int((df["exit_reason"] == "TIME").sum()),
    }])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="watchlist_module.py など（.py or .json）")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--tp", type=float, default=0.06)
    ap.add_argument("--sl", type=float, default=0.03)
    ap.add_argument("--hold-max", type=int, default=40)
    ap.add_argument("--priority", choices=["SL_first", "TP_first", "GapFair"], default="GapFair")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--outdir", default="backtest_out")
    ap.add_argument("--max-workers", type=int, default=3)  # ←同時3銘柄まで
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    wl = load_watchlist(args.watch)
    items = list(wl.items())[: args.limit]

    all_trades: List[Trade] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [
            ex.submit(
                backtest_one,
                code, name,
                args.start, args.end,
                args.tp, args.sl,
                args.hold_max,
                args.priority,
            )
            for code, name in items
        ]
        for fu in as_completed(futures):
            try:
                all_trades.extend(fu.result())
            except Exception as e:
                # 銘柄ごとの失敗は握りつぶして続行
                print("[WARN] worker error:", e)

    df_trades = trades_to_df(all_trades)
    df_sum = summary_df(df_trades)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join(args.outdir, f"trades_{stamp}.csv")
    summary_path = os.path.join(args.outdir, f"summary_{stamp}.csv")

    # 取引ゼロでも「空CSV」は保存する（Artifactsで確認できるように）
    df_trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    df_sum.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {trades_path}")
    print(f"Saved: {summary_path}")
    print(f"Trades: {len(df_trades)}")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
