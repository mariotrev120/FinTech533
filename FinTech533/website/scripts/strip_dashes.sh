#!/usr/bin/env bash
# Post-render: strip em/en dashes from docs/ (nuke any stragglers Quarto auto-insert).
# Replaces U+2014 (em dash) and U+2013 (en dash) with ASCII equivalents.
set -e
DOCS_DIR="${QUARTO_PROJECT_OUTPUT_DIR:-../../docs}"
cd "$DOCS_DIR" 2>/dev/null || cd "$(dirname "$0")/../../../docs"
# Use python to do unicode replace safely across all .html files
python3 - <<'PY'
import os, pathlib
root = pathlib.Path('.').resolve()
count = 0
for p in root.rglob('*.html'):
    try:
        s = p.read_text(encoding='utf-8', errors='replace')
    except Exception:
        continue
    new = s.replace('\u2014', '-').replace('\u2013', '|')
    if new != s:
        p.write_text(new, encoding='utf-8')
        count += 1
print(f'strip_dashes: cleaned {count} html files under {root}')
PY
