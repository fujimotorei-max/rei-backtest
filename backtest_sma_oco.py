import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime

# =========================
# yfinance MultiIndex対策
# =========================
def normalize_ohlc(df: pd.DataFrame, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(code, axis=1, level=1, drop_level=True)
        except Exception:
            df.columns = [c[0] for c in df.columns]

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]

    return df


def add_sma(df: pd.DataFrame, windows=(5, 25, 75)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(w).mean()
    return df


# =========================
# バックテスト本体
# =========================
def backtest(
    watchlist: dict,
    start: str,
    end: str,
    tp: float,
    sl: float,
    hold_max: int,
    max_positions: int,
):
    trades = []
    positions = []

    for code, name in watchlist.items():
        print(f"[LOAD] {code}")
        df = yf.download(
            code,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        df = normalize_ohlc(df, code)
        if df is None or df.empty:
            continue

        df = add_sma(df)
        df = df.dropna()
        if len(df) < 2:
            continue

        for i in range(1, len(df)):
            date = df.index[i]
            prev = df.iloc[i - 1]
            curr = df.iloc[i]

            # ===== 既存ポジションの決済判定 =====
            for pos in positions[:]:
                if pos["code"] != code:
                    continue

                days_held = (date - pos["entry_date"]).days
                hit_tp = curr["High"] >= pos["tp"]
                hit_sl = curr["Low"] <= pos["sl"]

                exit_price = None
                result = None

                if hit_tp:
                    exit_price = pos["tp"]
                    result = "TP"
                elif hit_sl:
                    exit_price = pos["sl"]
                    result = "SL"
                elif days_held >= hold_max:
                    exit_price = curr["Close"]
                    result = "TIME"

                if exit_price is not None:
                    trades.append({
                        "code": code,
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "result": result,
                        "return": (exit_price / pos["entry_price"]) - 1,
                    })
                    positions.remove(pos)

            # ===== エントリー判定 =====
            if len(positions) >= max_positions:
                continue

            already_holding = any(p["code"] == code for p in positions)
            if already_holding:
                continue

            # 厳しい条件
            gcross = (prev["SMA5"] <= prev["SMA25"]) and (curr["SMA5"] > curr["SMA25"])
            prev_align = prev["SMA5"] > prev["SMA25"] > prev["SMA75"]
            curr_align = curr["SMA5"] > curr["SMA25"] > curr["SMA75"]
            align_new = (not prev_align) and curr_align

            if gcross or align_new:
                entry_price = curr["Close"]
                positions.append({
                    "code": code,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "tp": entry_price * (1 + tp),
                    "sl": entry_price * (1 - sl),
                })

    return pd.DataFrame(trades)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tp", type=float, default=0.06)
    parser.add_argument("--sl", type=float, default=0.03)
    parser.add_argument("--hold-max", type=int, default=60)
    parser.add_argument("--max-pos", type=int, default=3)
    parser.add_argument("--outdir", default="backtest_out")

    args = parser.parse_args()

    watchlist = {}
    exec(open(args.watch, encoding="utf-8").read(), {}, watchlist)
    watchlist = watchlist["watchlist"]

    os.makedirs(args.outdir, exist_ok=True)

    trades = backtest(
        watchlist,
        args.start,
        args.end,
        args.tp,
        args.sl,
        args.hold_max,
        args.max_pos,
    )

    trades.to_csv(f"{args.outdir}/trades.csv", index=False)

    if len(trades) == 0:
        print("⚠️ trades = 0（条件が厳しすぎる or 期間が悪い）")
    else:
        print(trades.head())
        print(f"Total trades: {len(trades)}")
