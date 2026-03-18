#!/bin/bash
# Run the Excel and CSV update script
# Requires: pip install pandas openpyxl (or: pip install -r requirements.txt)

cd "$(dirname "$0")/.."
python3 -c "
import sys
try:
    import pandas as pd
    import openpyxl
except ImportError as e:
    print('Missing dependencies. Run: pip install pandas openpyxl')
    print('Or: pip install -r requirements.txt')
    sys.exit(1)
" || exit 1
python3 scripts/data_management/update_excel_and_csv.py
