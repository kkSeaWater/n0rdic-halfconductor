import os
import glob
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import io
from pathlib import Path
import re
import numpy as np  # Added for pie chart calculations

# ---------- Helpers ----------
def load_csv(path):
    # Read lines manually to skip summary section
    with open(path, 'r') as f:
        lines = f.readlines()
    data_lines = [line for line in lines if not line.strip().startswith('#')]
    df = pd.read_csv(io.StringIO(''.join(data_lines)))
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # Make timezone-naive for matplotlib
    
    # Convert numerics safely
    num_cols = [
        "rtt_ms", "lqi_in", "lqi_out", "age_s", "tx_total", "rx_total",
        "tx_err_cca", "tx_retry", "rx_err_fcs", "avg_current_uA"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "event" not in df.columns:
        df["event"] = ""
    if "state" not in df.columns:
        df["state"] = ""
    
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def split_events(event_str):
    if not isinstance(event_str, str) or not event_str:
        return set()
    return set(e.strip() for e in event_str.split("|") if e.strip())

def event_times(df, name):
    idx = [i for i, s in enumerate(df["event"]) if name in split_events(s)]
    return df.loc[idx, "timestamp"]

def mark_detached(ax, df):
    """Color background when state == detached."""
    in_detach = False
    start_time = None
    for _, row in df.iterrows():
        state = str(row.get("state", "")).lower()
        if not in_detach and state == "detached":
            start_time = row["timestamp"]
            in_detach = True
        elif in_detach and state != "detached":
            ax.axvspan(start_time, row["timestamp"], color="red", alpha=0.15)
            in_detach = False
    if in_detach and start_time is not None:
        ax.axvspan(start_time, df["timestamp"].iloc[-1], color="red", alpha=0.15)

def add_event_markers(ax, df):
    """Add vertical lines for key events."""
    events = {
        "detached_start": ("red", "Detach"),
        "reattached": ("green", "Reattach"),
        "parent_switch": ("blue", "Parent Switch")
    }
    for event, (color, label) in events.items():
        times = event_times(df, event)
        for t in times:
            ax.axvline(t, color=color, linestyle="--", alpha=0.5, label=label if event not in ax.get_legend_handles_labels()[1] else "")

def mark_parent_periods(ax, df):
    """Shade background for different parent attachment periods."""
    if "parent_rloc16" not in df.columns:
        return
    unique_parents = [p for p in df["parent_rloc16"].unique() if p != 'none']
    if not unique_parents:
        return  # No parents
    from matplotlib import cm
    parent_color = {p: cm.Paired(i / len(unique_parents)) for i, p in enumerate(unique_parents)}
    parent_color['none'] = 'lightgray'
    current_parent = None
    start_time = None
    for _, row in df.iterrows():
        parent = row["parent_rloc16"]
        if parent != current_parent:
            if current_parent is not None and start_time is not None:
                label_text = 'No Parent' if current_parent == 'none' else current_parent
                ax.axvspan(start_time, row["timestamp"], color=parent_color[current_parent], alpha=0.3, label=label_text if label_text not in ax.get_legend_handles_labels()[1] else "")
            current_parent = parent
            start_time = row["timestamp"]
    # Last period
    if current_parent is not None and start_time is not None:
        label_text = 'No Parent' if current_parent == 'none' else current_parent
        ax.axvspan(start_time, df["timestamp"].iloc[-1], color=parent_color[current_parent], alpha=0.3, label=label_text if label_text not in ax.get_legend_handles_labels()[1] else "")

def savefig(fig, csv_path, suffix):
    base = os.path.splitext(csv_path)[0]
    out = f"{base}_{suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved: {out}")

# Extract timestamp from filename
def extract_timestamp(filename):
    try:
        # Try to find date-time pattern in filename like YYYYMMDD_HHMMSS or similar
        parts = filename.stem.split('_')
        for part in parts:
            if len(part) == 8 and part.isdigit():  # YYYYMMDD
                next_part = parts[parts.index(part) + 1] if parts.index(part) + 1 < len(parts) else ''
                if len(next_part) == 6 and next_part.isdigit():  # HHMMSS
                    ts_str = f"{part}_{next_part}"
                    return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
        # Alternative pattern like 20251023T093856
        for part in filename.stem.split('-'):
            if 'T' in part:
                ts_str = part.replace('T', '_').split('_')[0] + '_' + part.split('T')[1]
                return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
        raise ValueError
    except (ValueError, IndexError):
        return datetime.fromtimestamp(filename.stat().st_mtime)

# Function to select file interactively
def select_file(files, file_type):
    print(f"Available {file_type} files:")
    for i, file in enumerate(files, 1):
        ts = extract_timestamp(file)
        print(f"{i}: {file.name} (timestamp: {ts})")
    
    while True:
        try:
            choice = int(input(f"Enter the number of the {file_type} file to use: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# ---------- Attachment Pie Functions ----------
def find_matching_txt(csv_path):
    stem = Path(csv_path).stem
    parts = stem.split('_')
    if len(parts) >= 3:
        dt_part = '_'.join(parts[-2:])
    else:
        dt_part = parts[-1]
    candidate_names = [
        f"risk_logs_{dt_part}.txt",
        f"risk_logs_{parts[-1]}.txt",
        f"child_log_{dt_part}.txt",
        f"child_log_{parts[-1]}.txt",
    ]
    for name in candidate_names:
        txt_path = Path(csv_path).parent / name
        if txt_path.exists():
            return txt_path
    return None

def parse_switch_durations(txt_path):
    if not txt_path:
        return []
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    switches = []
    for line in lines:
        match = re.search(
            r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s+event=(predictive_switch|parent_switch)\s+ok=(True|False)\s+elapsed=([\d\.]+)s', line)
        if match:
            time_str = match.group(1)
            ok = match.group(3) == 'True'
            elapsed = float(match.group(4))
            try:
                date_str = Path(txt_path).stem.split('_')[-2]
                datetime_str = f"{date_str} {time_str}"
                switch_time = datetime.strptime(datetime_str, '%Y%m%d %H:%M:%S.%f')
            except Exception:
                today = datetime.now().strftime('%Y%m%d')
                switch_time = datetime.strptime(f"{today} {time_str}", '%Y%m%d %H:%M:%S.%f')
            switches.append((switch_time, elapsed, ok))
    return switches

def compute_attachment_durations(df, switches=None):
    if df.empty or "timestamp" not in df.columns:
        return {}

    if "state" not in df.columns:
        df["state"] = ""
    if "parent_rloc16" not in df.columns:
        df["parent_rloc16"] = "none"

    df = df.copy()
    df["t_next"] = df["timestamp"].shift(-1)
    df["dt"] = (df["t_next"] - df["timestamp"]).dt.total_seconds()
    df = df[df["dt"] > 0]

    def label_row(row):
        st = str(row.get("state", "")).lower()
        parent = str(row.get("parent_rloc16", "none"))
        if st == "detached" or parent == "none" or parent.strip() == "":
            return "Switching"
        return f"Parent {parent}"

    df["label"] = df.apply(label_row, axis=1)
    durations = df.groupby("label")["dt"].sum().to_dict()

    if switches:
        txt_switch_total = sum(elapsed for (_, elapsed, _) in switches)
        durations["Switching"] = durations.get("Switching", 0.0) + txt_switch_total  # Add to existing detached time

    return durations

def plot_attachment_pie(durations, csv_path):
    if not durations:
        print("No durations to plot for attachment pie.")
        return None
    labels = list(durations.keys())
    values = [durations[k] for k in labels]
    
    def autopct_func(pct):
        total = sum(values)
        absolute = (pct / 100.) * total
        return "{:.1f}%\n({:.2f}s)".format(pct, absolute)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, autopct=autopct_func)
    ax.set_title("Attachment Time Distribution")
    savefig(fig, csv_path, "attachment_pie")
    return fig

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Analyze OpenThread telemetry CSV.")
    parser.add_argument("--folder", default=r"C:\Users\adire\Desktop\nordic_logs", help="Directory containing telemetry CSVs")
    args = parser.parse_args()
    
    folder = Path(args.folder)
    telemetry_files = [f for f in folder.glob('*telemetry_*.csv')]
    
    if not telemetry_files:
        raise FileNotFoundError(f"No telemetry CSV found in {folder}")
    
    telemetry_files.sort(key=extract_timestamp)
    
    selected_telemetry = select_file(telemetry_files, "telemetry")
    csv_path = str(selected_telemetry)
    print(f"Analyzing: {csv_path}")
    
    txt_path = find_matching_txt(csv_path)
    switches = parse_switch_durations(txt_path)
    
    df = load_csv(csv_path)
    t = df["timestamp"]
    has_power = "avg_current_uA" in df.columns and df["avg_current_uA"].notna().any()
    
    # 1) RTT over time
    if "rtt_ms" in df.columns and df["rtt_ms"].notna().any():
        fig1, ax = plt.subplots(figsize=(12, 6))
        mark_detached(ax, df)
        add_event_markers(ax, df)
        ax.plot(t, df["rtt_ms"], label="RTT (ms)")
        ax.set_title("RTT over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("RTT (ms)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        savefig(fig1, csv_path, "rtt")
    
    # 2) LQI In/Out
    if all(col in df for col in ["lqi_in", "lqi_out"]):
        fig2, ax = plt.subplots(figsize=(12, 6))
        mark_detached(ax, df)
        add_event_markers(ax, df)
        ax.plot(t, df["lqi_in"], label="LQI In")
        ax.plot(t, df["lqi_out"], label="LQI Out")
        ax.set_title("LQI In/Out over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("LQI")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        savefig(fig2, csv_path, "lqi")
    
    # 3) Age
    if "age_s" in df.columns:
        fig3, ax = plt.subplots(figsize=(12, 6))
        mark_detached(ax, df)
        add_event_markers(ax, df)
        ax.plot(t, df["age_s"], label="Age (s)")
        ax.set_title("Parent Age over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Age (s)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        savefig(fig3, csv_path, "age")
    
    # 4) MAC Counters (TX/RX totals, errors)
    if all(col in df for col in ["tx_total", "rx_total", "tx_err_cca", "tx_retry", "rx_err_fcs"]):
        fig4, ax1 = plt.subplots(figsize=(12, 6))
        mark_detached(ax1, df)
        add_event_markers(ax1, df)
        ax1.plot(t, df["tx_total"], label="TX Total")
        ax1.plot(t, df["rx_total"], label="RX Total")
        ax1.set_ylabel("Packets")
        ax1.legend(loc="upper left")
        
        ax2 = ax1.twinx()
        ax2.plot(t, df["tx_err_cca"], color="orange", label="TX Err CCA")
        ax2.plot(t, df["tx_retry"], color="purple", label="TX Retry")
        ax2.plot(t, df["rx_err_fcs"], color="brown", label="RX Err FCS")
        ax2.set_ylabel("Errors")
        ax2.legend(loc="upper right")
        
        ax1.set_title("MAC Counters over Time")
        ax1.set_xlabel("Timestamp")
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True)
        savefig(fig4, csv_path, "mac_counters")
    
    # 5) Power (absolute)
    if has_power:
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        mark_detached(ax5, df)
        add_event_markers(ax5, df)
        ax5.plot(t, df["avg_current_uA"], color="tab:red", label="Current (μA)", linewidth=2)
        # Tight y-limits to show variations better
        i_min, i_max = df["avg_current_uA"].min(), df["avg_current_uA"].max()
        ax5.set_ylim(i_min - 20, i_max + 20)
        ax5.set_title("Power over Time")
        ax5.set_xlabel("Timestamp")
        ax5.set_ylabel("Current (μA)")
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True)
        ax5.legend()
        savefig(fig5, csv_path, "power")
    elif has_power:
        warnings.warn("Skipping power plot: No valid data.")
    
    # 6) Transmission and Power (original, absolute current)
    if has_power and df["avg_current_uA"].notna().any() and all(col in df for col in ["tx_total", "rx_total"]):
        df['tx_delta'] = df['tx_total'].diff().fillna(0)
        df['rx_delta'] = df['rx_total'].diff().fillna(0)
        df['msg_activity'] = df['tx_delta'] + df['rx_delta']
        fig6, ax1 = plt.subplots(figsize=(12, 6))
        mark_detached(ax1, df)
        add_event_markers(ax1, df)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Absolute current on left axis
        ax1.plot(t, df["avg_current_uA"], color="tab:red", label="Current (μA)", linewidth=2)
        ax1.set_title("Message Transmission and Power")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Current (μA)", color="tab:red")
        ax1.tick_params(axis='y', labelcolor="tab:red")
        # Tight limits around absolute values
        i_min, i_max = df["avg_current_uA"].min(), df["avg_current_uA"].max()
        ax1.set_ylim(i_min - 50, i_max + 50)  # Adjust buffer as needed
        
        # Twin axis for message activity
        ax2 = ax1.twinx()
        # Compute bar width based on average time delta (in days for matplotlib)
        delta_t_days = (t - t.shift(1)).mean().total_seconds() / 86400 if len(t) > 1 else 0.001
        ax2.bar(t, df["msg_activity"], width=delta_t_days * 0.8, color="tab:blue", alpha=0.6, label="Msg Activity (Δ TX+RX)")
        ax2.set_ylabel("Message Activity (packets)", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")
        
        fig6.legend(loc="upper right")
        savefig(fig6, csv_path, "transmission_power")
    else:
        warnings.warn("Skipping transmission/power plot: Missing required columns (avg_current_uA, tx_total, rx_total).")
    
    # 7) Transmission and Power with Parent Periods (absolute current)
    if has_power and df["avg_current_uA"].notna().any() and all(col in df for col in ["tx_total", "rx_total"]) and "parent_rloc16" in df.columns:
        df['tx_delta'] = df['tx_total'].diff().fillna(0)
        df['rx_delta'] = df['rx_total'].diff().fillna(0)
        df['msg_activity'] = df['tx_delta'] + df['rx_delta']
        
        fig7, ax3 = plt.subplots(figsize=(12, 6))
        mark_detached(ax3, df)
        add_event_markers(ax3, df)
        mark_parent_periods(ax3, df)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # Absolute current
        ax3.plot(t, df["avg_current_uA"], color="tab:red", label="Current (μA)", linewidth=2)
        ax3.set_title("Message Transmission and Power with Parent Periods")
        ax3.set_xlabel("Timestamp")
        ax3.set_ylabel("Current (μA)", color="tab:red")
        ax3.tick_params(axis='y', labelcolor="tab:red")
        ax3.set_ylim(i_min - 50, i_max + 50)  # Adjust buffer as needed
        
        # Twin axis for message activity
        ax4 = ax3.twinx()
        ax4.bar(t, df["msg_activity"], width=delta_t_days * 0.8, color="tab:blue", alpha=0.6, label="Msg Activity (Δ TX+RX)")
        ax4.set_ylabel("Message Activity (packets)", color="tab:blue")
        ax4.tick_params(axis='y', labelcolor="tab:blue")
        
        fig7.legend(loc="upper right")
        savefig(fig7, csv_path, "transmission_power_with_parents")
    else:
        warnings.warn("Skipping transmission/power with parents plot: Missing required columns.")
    
    # ---------- Attachment Pie ----------
    durations = compute_attachment_durations(df, switches)
    total_s = sum(durations.values())
    if total_s > 0:
        print("\n--- Attachment Time Summary ---")
        for k, v in sorted(durations.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * v / total_s
            print(f"{k}: {v:.2f}s ({pct:.1f}%)")
        switching_s = durations.get("Switching", 0.0)
        print(f"Total Switching Time: {switching_s:.2f}s ({(switching_s/total_s*100):.1f}%)")
        plot_attachment_pie(durations, csv_path)
    else:
        print("No timing information to build attachment pie.")
    
    # ---------- Summary ----------
    print("\n--- Summary ---")
    # RTT stats
    valid_rtt = df["rtt_ms"].dropna()
    if len(valid_rtt):
        print(f"Average RTT: {valid_rtt.mean():.2f} ms")
        print(f"Min RTT: {valid_rtt.min():.2f} ms, Max RTT: {valid_rtt.max():.2f} ms")
    else:
        print("No RTT data found.")
    
    # Power stats (if available) - Uses math from docstring (e.g., assume V=3 for example calc)
    if has_power:
        valid_power = df["avg_current_uA"].dropna()
        if len(valid_power):
            i_avg = valid_power.mean() / 1e6  # Convert μA to A
            v_assumed = 3.0  # Typical for nRF; change if known
            p_avg = v_assumed * i_avg
            print(f"Average Current: {valid_power.mean():.2f} μA")
            print(f"Min Current: {valid_power.min():.2f} μA, Max Current: {valid_power.max():.2f} μA")
            print(f"Example Avg Power (at {v_assumed}V): {p_avg * 1000:.2f} mW")  # mW for readability
            # Approximate energy: Assume uniform intervals
            if len(t) > 1:
                total_time_s = (t.max() - t.min()).total_seconds()
                area_approx = valid_power.mean() * total_time_s / 1e6  # μA * s / 1e6 = amp-seconds (charge)
                energy_j = v_assumed * area_approx
                print(f"Approx Total Energy (area under curve * V): {energy_j:.4f} J over {total_time_s:.1f} s")
            # By state/event
            if "state" in df:
                print("\nAvg Current by State:")
                print(df.groupby("state")["avg_current_uA"].mean().round(2))
            if "event" in df:
                print("\nAvg Current by Event:")
                print(df.groupby("event")["avg_current_uA"].mean().round(2))
            # New: By parent
            if "parent_rloc16" in df:
                print("\nAvg Current by Parent:")
                print(df.groupby("parent_rloc16")["avg_current_uA"].mean().round(2))
        else:
            print("No power data found.")
    
    # Events and uptime
    n_detach = len(event_times(df, "detached_start"))
    n_re = len(event_times(df, "reattached"))
    n_switch = len(event_times(df, "parent_switch"))
    total = len(df)
    detached_rows = sum(str(s).lower() == "detached" for s in df["state"])
    uptime_pct = 100.0 * (1 - detached_rows / total) if total > 0 else 0
    print(f"Detachments: {n_detach}, Reattachments: {n_re}, Parent switches: {n_switch}")
    print(f"Uptime (child state) = {uptime_pct:.1f}%")
    
    plt.show()

if __name__ == "__main__":
    main()