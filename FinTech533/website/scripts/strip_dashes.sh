#!/usr/bin/env bash
# Post-render: strip fancy unicode dashes/minuses from docs/ (Quarto and Plotly
# auto-insert them; we want ASCII for consistency). Replaces:
#   U+2014 (em dash)     -> "-"
#   U+2013 (en dash)     -> "|"
#   U+2212 (minus sign)  -> "-"  (Plotly number formatting emits this)
set -e
DOCS_DIR="${QUARTO_PROJECT_OUTPUT_DIR:-../../docs}"
cd "$DOCS_DIR" 2>/dev/null || cd "$(dirname "$0")/../../../docs"
python3 - <<'PY'
import pathlib
root = pathlib.Path('.').resolve()
count = 0
for p in root.rglob('*.html'):
    try:
        s = p.read_text(encoding='utf-8', errors='replace')
    except Exception:
        continue
    new = (s.replace('\u2014', '-')
             .replace('\u2013', '|')
             .replace('\u2212', '-'))
    if new != s:
        p.write_text(new, encoding='utf-8')
        count += 1
print(f'strip_dashes: cleaned {count} html files under {root}')
PY
