#!/bin/bash
# Waits for TWS to accept a handshake, then runs the full HW5 pipeline.
# Exits 0 on success, nonzero on failure.
set -e

PY="/home/mht120/projects/FinTech533/Trading/.venv/bin/python"
HW5="/home/mht120/projects/FinTech533/FinTech533/Homeworks/HW5"
SITE="/home/mht120/projects/FinTech533/FinTech533/website"
QUARTO="/home/mht120/.local/quarto-install/opt/quarto/bin/quarto"

cd "$HW5"
while true; do
  out=$("$PY" - <<'EOF'
import shinybroker as sb, sys
try:
    r = sb.fetch_historical_data(
        contract=sb.Contract({'symbol':'NVDA','secType':'STK','exchange':'SMART','currency':'USD'}),
        endDateTime='20241231 23:59:59',
        durationStr='5 D', barSizeSetting='1 day', whatToShow='Trades',
        host='172.29.208.1', port=7497, client_id=300,
    )
    print('READY', len(r['hst_dta']))
    sys.exit(0)
except Exception as e:
    print('WAITING', type(e).__name__)
    sys.exit(1)
EOF
  ) || true
  if echo "$out" | grep -q '^READY'; then
    echo "TWS READY: $out"
    break
  fi
  sleep 10
done

echo "Executing notebook end-to-end..."
"$PY" -m jupyter nbconvert --to notebook --execute Mario_BreakoutStrategy.ipynb --inplace

echo "Rendering site..."
cd "$SITE"
QUARTO_PYTHON="$PY" "$QUARTO" render

echo "Committing + pushing..."
cd /home/mht120/projects/FinTech533
git add FinTech533/Homeworks/HW5/*.csv FinTech533/Homeworks/HW5/*.ipynb FinTech533/Homeworks/HW5/*.html docs/ 2>/dev/null || true
git commit -m "HW5: populate OOS backtest with live TWS data" || true
git push

echo "ALL DONE"
