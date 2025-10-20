import pandas as pd
import os
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import argparse
from dateutil.parser import parse
import tzlocal

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Correlate OpenThread telemetry with PPK power data")
parser.add_argument('--ppk-delay', type=float, default=1.0, help='PPK start delay in seconds')
parser.add_argument('--window-ms', type=int, default=1000, help='Correlation window in milliseconds')
parser.add_argument('--log-dir', type=str, default=r"C:\Users\adire\Desktop\nordic_logs", help='Directory containing log files')
args = parser.parse_args()

# Directory to scan
log_dir = Path(args.log_dir)

# Find telemetry and PPK files
telemetry_files = [f for f in log_dir.glob('tele*.csv')]
ppk_files = [f for f in log_dir.glob('ppk2*.csv')]

if not telemetry_files or not ppk_files:
    print("Could not find telemetry or PPK files.")
    exit(1)

# Extract timestamp from filename
def extract_timestamp(filename):
    try:
        ts_str = filename.stem.split('_')[-1]  # Assumes timestamp is last part before .csv
        return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
    except (ValueError, IndexError):
        return datetime.fromtimestamp(filename.stat().st_mtime)

# Pair files by closest timestamps
telemetry_files.sort(key=extract_timestamp)
ppk_files.sort(key=extract_timestamp)
latest_telemetry = telemetry_files[-1]
latest_ppk = ppk_files[-1]

# Check if files are from the same run (within 1 hour)
tele_ts = extract_timestamp(latest_telemetry)
ppk_ts = extract_timestamp(latest_ppk)
if abs((tele_ts - ppk_ts).total_seconds()) > 3600:
    print(f"Warning: Telemetry ({tele_ts}) and PPK ({ppk_ts}) files may not be from the same run.")

print(f"Correlating {latest_telemetry} with {latest_ppk}")

# Load Telemetry CSV
df_telemetry = pd.read_csv(latest_telemetry, encoding='utf-8-sig')
df_telemetry['timestamp'] = pd.to_datetime(df_telemetry['timestamp'])
df_telemetry.set_index('timestamp', inplace=True)
telemetry_start_time = df_telemetry.index.min()

# Read script start time from corresponding .txt file
log_txt_path = latest_telemetry.with_suffix('.txt')
try:
    with open(log_txt_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line.startswith("# Script started at "):
            script_start_time_str = first_line.replace("# Script started at ", "")
            script_start_time = parse(script_start_time_str)  # Robust parsing
            script_start_time = tzlocal.get_localzone().localize(script_start_time)  # Use local timezone
        else:
            script_start_time = telemetry_start_time  # Fallback
except FileNotFoundError:
    print(f"Warning: {log_txt_path} not found, using telemetry start time.")
    script_start_time = telemetry_start_time

# Load PPK CSV and validate columns
required_cols = ['Timestamp(ms)', 'Current(uA)']
df_ppk = pd.read_csv(latest_ppk)
if not all(col in df_ppk.columns for col in required_cols):
    print(f"PPK CSV missing required columns: {required_cols}")
    exit(1)

# Align PPK timestamps
ppk_start = script_start_time + timedelta(seconds=args.ppk_delay)
ppk_first_timestamp = df_ppk['Timestamp(ms)'].iloc[0]
ppk_start = ppk_start + pd.to_timedelta(ppk_first_timestamp, unit='ms')
df_ppk['absolute_time'] = ppk_start + pd.to_timedelta(df_ppk['Timestamp(ms)'] - ppk_first_timestamp, unit='ms')
df_ppk.set_index('absolute_time', inplace=True)

# Check and trim to overlap
ppk_start_time = df_ppk.index.min()
ppk_end_time = df_ppk.index.max()
telemetry_end_time = df_telemetry.index.max()
print(f"Script start time: {script_start_time}")
print(f"PPK assumed start: {ppk_start}")
print(f"PPK range: {ppk_start_time} to {ppk_end_time}")
print(f"Telemetry range: {telemetry_start_time} to {telemetry_end_time}")

overlap_start = max(ppk_start_time, telemetry_start_time)
overlap_end = min(ppk_end_time, telemetry_end_time)
if overlap_start > overlap_end:
    print(f"No overlap detected. PPK range: {ppk_start_time} to {ppk_end_time}, Telemetry range: {telemetry_start_time} to {telemetry_end_time}")
    exit(1)

df_ppk = df_ppk.loc[overlap_start:overlap_end]
df_telemetry = df_telemetry.loc[overlap_start:overlap_end]

# Function to average current around a telemetry timestamp
def get_avg_current(timestamp, window_ms=args.window_ms):
    start = timestamp - pd.Timedelta(milliseconds=window_ms)
    end = timestamp + pd.Timedelta(milliseconds=window_ms)
    window_data = df_ppk[(df_ppk.index >= start) & (df_ppk.index <= end)]
    if not window_data.empty:
        return window_data['Current(uA)'].mean()
    return None

# Add average current to telemetry DF
df_telemetry['avg_current_uA'] = df_telemetry.index.map(get_avg_current)

# Calculate summary statistics
summary = df_telemetry.groupby('event')['avg_current_uA'].mean().round(2)
summary_str = "Average current by event:\n" + summary.to_string()

# Output correlated telemetry to a new CSV
tz_cest = pytz.timezone('Europe/Paris')
output_path = log_dir / f"correlated_telemetry_{datetime.now(tz_cest).strftime('%Y%m%d_%H%M%S')}.csv"
df_telemetry.to_csv(output_path)
with open(output_path, 'a', encoding='utf-8') as f:
    f.write("\n\n# Summary Statistics\n")
    f.write(summary_str)
print(f"Correlated logs saved to: {output_path}")
print(summary_str)