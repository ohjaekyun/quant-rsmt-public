#!/bin/zsh
# Weekly forward recording (docs/MODELING.md 26.2, docs/HOLDOUT.md): run once a week, e.g. Monday morning KST.
#   - forward-holdout snapshots of the served signals (KR, US) and scoring of matured snapshots
#   - API-key ledgers (Finnhub, Naver) - these are daily; this weekly run keeps them going if cron is weekly
#   - US v9 forward picks: scored once their 20-session labels exist
#   - score-vs-volatility monitor of the served models (docs/MODELING.md 56.2)
# Monthly (first run of the month): refresh the US v9 model and record new picks.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG_DIR=.cache/quantcio/logs/forward  # git-ignored
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d)
{
  echo "== $(date) weekly forward"
  .venv/bin/quantcio track snapshot --market KR
  .venv/bin/quantcio track snapshot --market US
  .venv/bin/quantcio track score
  .venv/bin/python scripts/serve_avoid.py  # 105: US avoid-side reference flag of the report (display only)
  .venv/bin/python scripts/api_keys.py ledger
  .venv/bin/python scripts/us_v9_forward.py
  .venv/bin/python scripts/paper_trading.py report
  .venv/bin/python scripts/risk_overlap_study.py record  # rank corr(score, volatility) of the served models (56.2)
  for w in naver_search naver_shop wiki_top us_soc census nyt; do  # 69: refresh the daily / monthly series
    .venv/bin/python scripts/v28_popular_data.py fetch --what "$w"
  done
  .venv/bin/python scripts/v29_credit_vc_data.py fetch --what all  # 70: credit, loans, Form D, KR VC ledger
  .venv/bin/python scripts/v31_macro_data.py fetch                # 75: rates (full history), prices by item, production
  .venv/bin/python scripts/v32_trade_data.py fetch                # 76: customs exports / imports by product (last 2 years)
  if [ "$(date +%d)" -le 7 ]; then
    echo "== monthly: refresh the US v9 model and record new picks"
    .venv/bin/python scripts/us_v9_serve.py
  fi
  echo "== done $(date)"
} >> "$LOG_DIR/weekly_$STAMP.log" 2>&1
