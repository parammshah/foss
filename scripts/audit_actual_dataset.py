import os
import hashlib
import datetime
import pandas as pd
import numpy as np

path = '/Users/paramshah/Downloads/treasury_4tier_dataset.csv'

# 1. Identity
st = os.stat(path)
with open(path, 'rb') as f:
    sha = hashlib.sha256(f.read()).hexdigest()

print('=== FILE IDENTITY ===')
print('Resolved Absolute Path:', os.path.abspath(path))
print('File Size (bytes):', st.st_size)
print('MTime (UTC):', datetime.datetime.fromtimestamp(st.st_mtime, tz=datetime.timezone.utc).isoformat())
print('SHA-256 Hash:', sha)

# Load raw without preprocessor
raw_df = pd.read_csv(path)
print('Row Count:', len(raw_df))
print('Column Count:', len(raw_df.columns))
print('Raw Columns:', raw_df.columns.tolist())

# 2. First 20 rows
print('\n=== FIRST 20 RAW ROWS ===')
print(raw_df[['timestamp', 'deposits_usd', 'withdrawals_usd', 'net_flow_usd']].head(20).to_string())

# 3. Statistics
print('\n=== CASH FLOW STATISTICS (RAW) ===')
for col in ['deposits_usd', 'withdrawals_usd', 'net_flow_usd']:
    s = raw_df[col]
    print(f'-- {col} --')
    print(f'Min:    {s.min():,.2f}')
    print(f'Mean:   {s.mean():,.2f}')
    print(f'Median: {s.median():,.2f}')
    print(f'Std:    {s.std():,.2f}')
    print(f'p95:    {s.quantile(0.95):,.2f}')
    print(f'Max:    {s.max():,.2f}')

# Extremes and timestamps
min_dep_idx = raw_df['deposits_usd'].idxmin()
max_dep_idx = raw_df['deposits_usd'].idxmax()
min_wth_idx = raw_df['withdrawals_usd'].idxmin()
max_wth_idx = raw_df['withdrawals_usd'].idxmax()

print('\n=== EXTREMES & TIMESTAMPS ===')
print(f'Min Deposits:    ${raw_df.loc[min_dep_idx, "deposits_usd"]:,.2f} at {raw_df.loc[min_dep_idx, "timestamp"]}')
print(f'Max Deposits:    ${raw_df.loc[max_dep_idx, "deposits_usd"]:,.2f} at {raw_df.loc[max_dep_idx, "timestamp"]}')
print(f'Min Withdrawals: ${raw_df.loc[min_wth_idx, "withdrawals_usd"]:,.2f} at {raw_df.loc[min_wth_idx, "timestamp"]}')
print(f'Max Withdrawals: ${raw_df.loc[max_wth_idx, "withdrawals_usd"]:,.2f} at {raw_df.loc[max_wth_idx, "timestamp"]}')

# Total sums
print('\n=== TOTAL CASH FLOW SUMS ===')
print(f'Total Deposits:    ${raw_df["deposits_usd"].sum():,.2f}')
print(f'Total Withdrawals: ${raw_df["withdrawals_usd"].sum():,.2f}')
print(f'Total Net Flow:    ${raw_df["net_flow_usd"].sum():,.2f}')

# Net flow reconciliation
diff = (raw_df['deposits_usd'] - raw_df['withdrawals_usd']) - raw_df['net_flow_usd']
abs_diff = diff.abs()
print('\n=== NET FLOW RECONCILIATION ===')
print(f'Max absolute discrepancy: ${abs_diff.max():.6f}')
print(f'Mean absolute discrepancy: ${abs_diff.mean():.6f}')
print(f'Discrepancies > $0.01: {(abs_diff > 0.010001).sum()}')
print(f'Discrepancies > $0.005: {(abs_diff > 0.005).sum()}')
print(f'Discrepancies > $1.00: {(abs_diff > 1.00).sum()}')
