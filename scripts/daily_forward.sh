#!/bin/zsh
# Daily forward recording (docs/MODELING.md 42, 50, 51, 64): US option chains, the paper-trading book, US analyst
# estimates, the remaining OpenDART items and KR news headlines (Naver search API). Run once a day,
# e.g. 07:00 KST Tuesday to Saturday (after the US close). launchd runs a missed run after the Mac wakes up.
set -u
cd "$(dirname "$0")/.." || exit 1
LOG_DIR=.cache/quantcio/logs/forward  # git-ignored
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d)
{
  echo "== $(date) daily forward"
  .venv/bin/python scripts/us_options_record.py
  .venv/bin/python scripts/paper_trading.py run
  .venv/bin/python scripts/v18_data.py ledger                    # US analyst estimates (docs/MODELING.md 51)
  .venv/bin/python scripts/v18_data.py fetch --what dart_extra   # resumes until complete (daily OpenDART quota)
  .venv/bin/python scripts/api_keys.py ledger --only Naver       # KR headline text -> cache (docs/MODELING.md 64)
  for w in youtube kobis library koneps; do                     # 69: YouTube chart (forward) + keyed backfills (skip w/o key)
    .venv/bin/python scripts/v28_popular_data.py fetch --what "$w"
  done
  echo "== done $(date)"
} >> "$LOG_DIR/daily_$STAMP.log" 2>&1
