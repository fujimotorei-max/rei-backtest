# SMA(5/25/75) + OCO Backtest (GitHub Actions)

## セットアップ
1. このフォルダを GitHub に新規リポジトリとしてアップロード
2. `watchlist_module.py` にあなたの巨大ウォッチリスト dict をそのまま貼り付け
3. Actions タブ → `Backtest SMA & OCO` を `Run workflow` から起動

## 実行パラメータ
- start, end（YYYY-MM-DD）
- limit（テスト数の上限）
- tp / sl（+6% / -3% 等）
- hold_max（最大保有日数）
- priority（TP_first / SL_first / GapFair）

実行後、`Artifacts` から `trades.csv` と `summary.csv` をダウンロードできます。