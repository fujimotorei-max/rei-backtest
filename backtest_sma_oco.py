
import argparse
import pandas as pd
import numpy as np
import importlib.util
import os
import datetime as dt

def load_watchlist(module_path: str):
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("watchlist_module", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "watchlist")
    elif module_path.endswith(".json"):
        import json
        with open(module_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported watchlist format. Use .py or .json")

def compute_sma(df, windows=(5,25,75)):
    out = df.copy()
    for w in windows:
        out[f"SMA{w}"] = out["Close"].rolling(w).mean()
    return out.dropna()

def day_rows(df):
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            raise ValueError("Missing OHLC column in data")
    return df

def decide_exit(row, entry, tp_rate, sl_rate, priority="GapFair"):
    tp = entry*(1+tp_rate)
    sl = entry*(1-sl_rate)
    o,h,l = row["Open"], row["High"], row["Low"]
    if o >= tp and o > entry: return o, "TP(gap)"
    if o <= sl and o < entry: return o, "SL(gap)"
    hit_tp = h >= tp
    hit_sl = l <= sl
    if hit_tp and hit_sl:
        if priority=="TP_first": return tp, "TP(first)"
        if priority=="SL_first": return sl, "SL(first)"
        if abs(o-tp) <= abs(o-sl): return tp, "TP(fair)"
        else: return sl, "SL(fair)"
    if hit_tp: return tp, "TP"
    if hit_sl: return sl, "SL"
    return None, None

def run_symbol(ticker, name, start, end, tp, sl, hold_max, priority):
    import yfinance as yf
    raw = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if raw is None or raw.empty or len(raw)<150:
        return []

    df = day_rows(raw)
    df = compute_sma(df)

    trades = []
    status = "FLAT"
    entry_px = None
    entry_idx = None

    idx = df.index

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        if status=="FLAT":
            gcross = (prev["SMA5"] <= prev["SMA25"]) and (curr["SMA5"] > curr["SMA25"])
            prev_align = (prev["SMA5"] > prev["SMA25"] > prev["SMA75"])
            curr_align = (curr["SMA5"] > curr["SMA25"] > curr["SMA75"])
            align_new  = (not prev_align) and curr_align
            if gcross or align_new:
                status="HOLD"
                entry_px = float(curr["Close"])
                entry_idx = i
        else:
            hold_days = i - entry_idx
            price, tag = decide_exit(curr, entry_px, tp, sl, priority)
            close_trade = False
            exit_px = None
            if price is not None:
                exit_px = float(price); close_trade=True
            elif hold_days >= hold_max:
                exit_px = float(curr["Close"]); tag = "TimeExit"; close_trade=True

            if close_trade:
                ret = (exit_px/entry_px - 1.0)
                trades.append({
                    "code": ticker,
                    "name": name,
                    "entry_date": str(idx[entry_idx].date()),
                    "entry_price": round(entry_px, 4),
                    "exit_date": str(idx[i].date()),
                    "exit_price": round(exit_px, 4),
                    "days_held": hold_days,
                    "exit_reason": tag,
                    "return": ret
                })
                status="FLAT"; entry_px=None; entry_idx=None

    return trades

def summarize(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return df, pd.DataFrame()
    df["win"] = df["return"] > 0
    by_code = df.groupby(["code","name"]).agg(
        trades=("return","count"),
        win_rate=("win","mean"),
        mean_ret=("return","mean"),
        median_ret=("return","median"),
        avg_days=("days_held","mean")
    ).reset_index()
    overall = pd.DataFrame([{
        "code":"__ALL__", "name":"ALL",
        "trades": int(by_code["trades"].sum()),
        "win_rate": float(df["win"].mean()),
        "mean_ret": float(df["return"].mean()),
        "median_ret": float(df["return"].median()),
        "avg_days": float(df["days_held"].mean())
    }])
    return df, pd.concat([by_code, overall], ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="watchlist_module.py or .json")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=False, default=None)
    ap.add_argument("--tp", type=float, default=0.06)
    ap.add_argument("--sl", type=float, default=0.03)
    ap.add_argument("--hold-max", type=int, default=40)
    ap.add_argument("--priority", choices=["TP_first","SL_first","GapFair"], default="GapFair")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--outdir", default="backtest_out")
    args = ap.parse_args()

    wl = load_watchlist(args.watch)
    items = list(wl.items())[:args.limit]
    all_trades = []
    for code, name in items:
        try:
            t = run_symbol(code, name, args.start, args.end, args.tp, args.sl, args.hold_max, args.priority)
            all_trades.extend(t)
        except Exception as e:
            print(f"[WARN] {code} error: {e}")

    os.makedirs(args.outdir, exist_ok=True)
    trades_df, summary_df = summarize(all_trades)
    trades_path = os.path.join(args.outdir, "trades.csv")
    summary_path = os.path.join(args.outdir, "summary.csv")
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if trades_df.empty:
        print("No trades produced.")
    else:
        print(f"Saved {trades_path} and {summary_path}")

if __name__ == "__main__":
    main()
