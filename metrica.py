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
import numpy as np  # For pie chart calculations if needed

# ---------- Helpers ----------
def load_csv(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    data_lines = [line for line in lines if not line.strip().startswith('#')]
    df = pd.read_csv(io.StringIO(''.join(data_lines)))

    if "timestamp" not in df.columns:
        for cand in ["time", "Timestamp", "Time"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)

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
    if "parent_rloc16" not in df.columns:
        df["parent_rloc16"] = "none"

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def parse_note(note):
        if pd.isna(note) or not isinstance(note, str):
            return pd.Series({'lqiMean': None, 'lqiStd': None, 'rttMean': None, 'cooldown': None})
        parts = note.split(';')
        parsed = {}
        for part in parts:
            if '=' in part:
                key, val = part.split('=')
                key = key.strip(); val = val.strip()
                try:
                    parsed[key] = float(val) if val else None
                except ValueError:
                    parsed[key] = None
        return pd.Series({
            'lqiMean': parsed.get('lqiMean'),
            'lqiStd': parsed.get('lqiStd'),
            'rttMean': parsed.get('rttMean'),
            'cooldown': parsed.get('cooldown')
        })

    if 'note' in df.columns:
        note_df = df['note'].apply(parse_note)
        df = pd.concat([df, note_df], axis=1)

    return df

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

def split_events(event_str):
    if not isinstance(event_str, str) or not event_str:
        return set()
    return set(e.strip() for e in event_str.split("|") if e.strip())

def event_times(df, name):
    idx = [i for i, s in enumerate(df["event"]) if name in split_events(s)]
    return df.loc[idx, "timestamp"]

def mark_detached(ax, df):
    in_detach = False
    start_time = None
    for _, row in df.iterrows():
        state = str(row.get("state", "")).lower()
        if not in_detach and state == "detached":
            start_time = row["timestamp"]
            in_detach = True
        elif in_detach and state != "detached":
            ax.axvspan(start_time, row["timestamp"], alpha=0.15)
            in_detach = False
    if in_detach and start_time is not None:
        ax.axvspan(start_time, df["timestamp"].iloc[-1], alpha=0.15)

def add_event_markers(ax, df):
    events = {
        "detached_start": ("Detach"),
        "reattached": ("Reattach"),
        "parent_switch": ("Parent Switch"),
        "predictive_switch": ("Predictive Switch")
    }
    for event, label in events.items():
        times = event_times(df, event)
        for t in times:
            ax.axvline(t, linestyle="--", alpha=0.5,
                       label=label if event not in ax.get_legend_handles_labels()[1] else "")

def mark_parent_periods(ax, df):
    if "parent_rloc16" not in df.columns:
        return
    unique_parents = [p for p in df["parent_rloc16"].unique() if p != 'none']
    if not unique_parents:
        return
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
                ax.axvspan(start_time, row["timestamp"],
                           color=parent_color.get(current_parent, None), alpha=0.3,
                           label=label_text if current_parent not in ax.get_legend_handles_labels()[1] else "")
            start_time = row["timestamp"]
            current_parent = parent
    if current_parent is not None and start_time is not None:
        label_text = 'No Parent' if current_parent == 'none' else current_parent
        ax.axvspan(start_time, df["timestamp"].iloc[-1],
                   color=parent_color.get(current_parent, None), alpha=0.3,
                   label=label_text if current_parent not in ax.get_legend_handles_labels()[1] else "")

def mark_switch_durations(ax, switches):
    for switch_time, elapsed, ok in switches:
        end_switch = switch_time + timedelta(seconds=elapsed)
        ax.axvspan(switch_time, end_switch, alpha=0.3)
        mid = switch_time + timedelta(seconds=elapsed / 2)
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"{elapsed:.2f}s",
                horizontalalignment='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))

def savefig(fig, csv_path, suffix, save_plots):
    if not save_plots:
        return
    stem = Path(csv_path).stem
    out_path = Path(csv_path).parent / f"{stem}_{suffix}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")

def zoom_to_switch(df, switches, window_sec=30):
    if not switches:
        warnings.warn("No switch duration found in TXT for zoom plot.")
        switch_time = event_times(df, "predictive_switch")
        if switch_time.empty:
            switch_time = event_times(df, "parent_switch")
        if switch_time.empty:
            return pd.DataFrame()
        switch_time = switch_time.iloc[0]
    else:
        switch_time = switches[0][0]
    start = switch_time - timedelta(seconds=window_sec)
    end = switch_time + timedelta(seconds=window_sec)
    return df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

# ---------- Attachment Pie ----------
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
        durations["Switching"] = max(durations.get("Switching", 0.0), txt_switch_total)

    return durations

def plot_attachment_pie(durations, csv_path, save_plots):
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
    savefig(fig, csv_path, "attachment_pie", save_plots)
    return fig

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

def analyze_telemetry(csv_path, save_plots):
    print(f"Analyzing: {csv_path}")
    txt_path = find_matching_txt(csv_path)
    switches = parse_switch_durations(txt_path)
    df = load_csv(csv_path)
    print("Unique parents:", df["parent_rloc16"].unique())

    t = df["timestamp"]
    has_power = "avg_current_uA" in df.columns and df["avg_current_uA"].notna().any()

    # 1) RTT over time
    if "rtt_ms" in df.columns and df["rtt_ms"].notna().any():
        fig1, ax = plt.subplots(figsize=(12, 6))
        mark_detached(ax, df)
        add_event_markers(ax, df)
        mark_switch_durations(ax, switches)
        ax.plot(t, df["rtt_ms"], label="RTT (ms)")
        ax.set_title("RTT over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("RTT (ms)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        savefig(fig1, csv_path, "rtt", save_plots)

    # 2) LQI In/Out
    if all(col in df for col in ["lqi_in", "lqi_out"]):
        fig2, ax = plt.subplots(figsize=(12, 6))
        mark_detached(ax, df)
        add_event_markers(ax, df)
        mark_switch_durations(ax, switches)
        ax.plot(t, df["lqi_in"], label="LQI In")
        ax.plot(t, df["lqi_out"], label="LQI Out")
        ax.set_title("LQI In/Out over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("LQI")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        savefig(fig2, csv_path, "lqi", save_plots)

    # 3) Age
    if "age_s" in df.columns:
        fig3, ax = plt.subplots(figsize=(12, 6))
        mark_detached(ax, df)
        add_event_markers(ax, df)
        mark_switch_durations(ax, switches)
        ax.plot(t, df["age_s"], label="Age (s)")
        ax.set_title("Parent Age over Time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Age (s)")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        savefig(fig3, csv_path, "age", save_plots)

    # 4) MAC Counters (TX/RX totals, errors)
    if all(col in df for col in ["tx_total", "rx_total", "tx_err_cca", "tx_retry", "rx_err_fcs"]):
        fig4, ax1 = plt.subplots(figsize=(12, 6))
        mark_detached(ax1, df)
        add_event_markers(ax1, df)
        mark_switch_durations(ax1, switches)
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
        savefig(fig4, csv_path, "mac_counters", save_plots)

    # 5) Power (absolute)
    if has_power:
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        mark_detached(ax5, df)
        add_event_markers(ax5, df)
        mark_switch_durations(ax5, switches)
        ax5.plot(t, df["avg_current_uA"], color="tab:red", label="Current (μA)", linewidth=2)
        i_min, i_max = df["avg_current_uA"].min(), df["avg_current_uA"].max()
        ax5.set_ylim(i_min - 20, i_max + 20)
        ax5.set_title("Power over Time")
        ax5.set_xlabel("Timestamp")
        ax5.set_ylabel("Current (μA)")
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True)
        ax5.legend()
        savefig(fig5, csv_path, "power", save_plots)

    # 6) Transmission and Power (original, absolute current)
    if has_power and df["avg_current_uA"].notna().any() and all(col in df for col in ["tx_total", "rx_total"]):
        df['tx_delta'] = df['tx_total'].diff().fillna(0)
        df['rx_delta'] = df['rx_total'].diff().fillna(0)
        df['msg_activity'] = df['tx_delta'] + df['rx_delta']
        
        fig6, ax1 = plt.subplots(figsize=(12, 6))
        mark_detached(ax1, df)
        add_event_markers(ax1, df)
        mark_switch_durations(ax1, switches)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        ax1.plot(t, df["avg_current_uA"], color="tab:red", label="Current (μA)", linewidth=2)
        ax1.set_title("Message Transmission and Power")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Current (μA)", color="tab:red")
        ax1.tick_params(axis='y', labelcolor="tab:red")
        i_min, i_max = df["avg_current_uA"].min(), df["avg_current_uA"].max()
        ax1.set_ylim(i_min - 50, i_max + 50)
        
        ax2 = ax1.twinx()
        delta_t_days = (t - t.shift(1)).mean().total_seconds() / 86400 if len(t) > 1 else 0.001
        ax2.bar(t, df["msg_activity"], width=delta_t_days * 0.8, color="tab:blue", alpha=0.6, label="Msg Activity (Δ TX+RX)")
        ax2.set_ylabel("Message Activity (packets)", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")
        
        fig6.legend(loc="upper right")
        savefig(fig6, csv_path, "transmission_power", save_plots)

    # 7) Transmission and Power with Parent Periods (absolute current)
    if has_power and df["avg_current_uA"].notna().any() and all(col in df for col in ["tx_total", "rx_total"]) and "parent_rloc16" in df.columns:
        df['tx_delta'] = df['tx_total'].diff().fillna(0)
        df['rx_delta'] = df['rx_total'].diff().fillna(0)
        df['msg_activity'] = df['tx_delta'] + df['rx_delta']
        
        fig7, ax3 = plt.subplots(figsize=(12, 6))
        mark_detached(ax3, df)
        add_event_markers(ax3, df)
        mark_parent_periods(ax3, df)
        mark_switch_durations(ax3, switches)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        ax3.plot(t, df["avg_current_uA"], color="tab:red", label="Current (μA)", linewidth=2)
        ax3.set_title("Message Transmission and Power with Parent Periods")
        ax3.set_xlabel("Timestamp")
        ax3.set_ylabel("Current (μA)", color="tab:red")
        ax3.tick_params(axis='y', labelcolor="tab:red")
        ax3.set_ylim(i_min - 50, i_max + 50)
        
        ax4 = ax3.twinx()
        ax4.bar(t, df["msg_activity"], width=delta_t_days * 0.8, color="tab:blue", alpha=0.6, label="Msg Activity (Δ TX+RX)")
        ax4.set_ylabel("Message Activity (packets)", color="tab:blue")
        ax4.tick_params(axis='y', labelcolor="tab:blue")
        
        fig7.legend(loc="upper right")
        savefig(fig7, csv_path, "transmission_power_with_parents", save_plots)

    # Attachment pie and summary
    durations = compute_attachment_durations(df, switches)
    total_s = sum(durations.values())
    if total_s > 0:
        print("\n--- Attachment Time Summary ---")
        for k, v in sorted(durations.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * v / total_s
            print(f"{k}: {v:.2f}s ({pct:.1f}%)")
        switching_s = durations.get("Switching", 0.0)
        print(f"Total Switching Time: {switching_s:.2f}s ({(switching_s/total_s*100):.1f}%)")
        plot_attachment_pie(durations, csv_path, save_plots)
    else:
        print("No timing information to build attachment pie.")

    # Summary
    print("\n--- Summary ---")
    metrics = {}
    valid_rtt = df["rtt_ms"].dropna()
    if len(valid_rtt):
        metrics["Avg RTT (ms)"] = round(valid_rtt.mean(), 2)
        metrics["Min RTT (ms)"] = round(valid_rtt.min(), 2)
        metrics["Max RTT (ms)"] = round(valid_rtt.max(), 2)
        print(f"Average RTT: {metrics['Avg RTT (ms)']} ms")
        print(f"Min RTT: {metrics['Min RTT (ms)']} ms, Max RTT: {metrics['Max RTT (ms)']} ms")
    else:
        print("No RTT data found.")
    
    total_time_s = (t.max() - t.min()).total_seconds() if len(t) > 1 else 0
    metrics["Total Time (s)"] = round(total_time_s, 1)

    if has_power:
        valid_power = df["avg_current_uA"].dropna()
        if len(valid_power):
            i_avg = valid_power.mean() / 1e6
            v_assumed = 3.0
            p_avg = v_assumed * i_avg
            metrics["Avg Current (μA)"] = round(valid_power.mean(), 2)
            metrics["Min Current (μA)"] = round(valid_power.min(), 2)
            metrics["Max Current (μA)"] = round(valid_power.max(), 2)
            metrics["Avg Power (mW)"] = round(p_avg * 1000, 2)
            print(f"Average Current: {metrics['Avg Current (μA)']} μA")
            print(f"Min Current: {metrics['Min Current (μA)']} μA, Max Current: {metrics['Max Current (μA)']} μA")
            print(f"Example Avg Power (at {v_assumed}V): {metrics['Avg Power (mW)']} mW")
            if len(t) > 1:
                area_approx = valid_power.mean() * total_time_s / 1e6
                energy_j = v_assumed * area_approx
                metrics["Total Energy (J)"] = round(energy_j, 4)
                print(f"Approx Total Energy (area under curve * V): {metrics['Total Energy (J)']} J over {total_time_s:.1f} s")
            if "state" in df:
                print("\nAvg Current by State:")
                print(df.groupby("state")["avg_current_uA"].mean().round(2))
            if "event" in df:
                print("\nAvg Current by Event:")
                print(df.groupby("event")["avg_current_uA"].mean().round(2))
            if "parent_rloc16" in df:
                print("\nAvg Current by Parent:")
                print(df.groupby("parent_rloc16")["avg_current_uA"].mean().round(2))
        else:
            print("No power data found.")
    
    n_detach = len(event_times(df, "detached_start"))
    n_re = len(event_times(df, "reattached"))
    n_switch = len(event_times(df, "parent_switch"))
    total = len(df)
    detached_rows = sum(str(s).lower() == "detached" for s in df["state"])
    uptime_pct = 100.0 * (1 - detached_rows / total) if total > 0 else 0
    metrics["Detachments"] = n_detach
    metrics["Reattachments"] = n_re
    metrics["Parent Switches"] = n_switch
    metrics["Uptime (%)"] = round(uptime_pct, 1)
    print(f"Detachments: {n_detach}, Reattachments: {n_re}, Parent switches: {n_switch}")
    print(f"Uptime (child state) = {uptime_pct:.1f}%")

    metrics["Switching Time (s)"] = round(switching_s, 2) if 'switching_s' in locals() else 0
    metrics["Switching (%)"] = round((switching_s / total_s * 100) if total_s > 0 else 0, 1)

    return metrics, df

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Analyze OpenThread telemetry CSV with risk metrics.")
    parser.add_argument("--folder", type=str, default=r"C:\Users\adire\Desktop\nordic_logs",
                        help="Folder with telemetry CSV")
    args = parser.parse_args()

    # Ask user if to save screenshots
    while True:
        response = input("Save screenshots? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            save_plots = response == 'y'
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    folder = Path(args.folder)
    telemetry_glob1 = glob.glob(os.path.join(folder, "correlated_risk_telemetry_*.csv"))
    telemetry_glob2 = glob.glob(os.path.join(folder, "risk_telemetry_*.csv"))
    telemetry_glob3 = glob.glob(os.path.join(folder, "*telemetry_*.csv"))
    telemetry_files = [Path(f) for f in set(telemetry_glob1 + telemetry_glob2 + telemetry_glob3)]
    
    if not telemetry_files:
        raise FileNotFoundError(f"No telemetry CSV found in {folder}")

    telemetry_files.sort(key=extract_timestamp)

    selected_telemetry1 = select_file(telemetry_files, "telemetry")
    metrics1, df1 = analyze_telemetry(selected_telemetry1, save_plots)
    file1_name = selected_telemetry1.stem

    # Ask if want to compare
    while True:
        response = input("Do you want to compare with another telemetry file? (y/n): ").strip().lower()
        if response in ['y', 'n']:
            compare = response == 'y'
            break
        print("Invalid input. Please enter 'y' or 'n'.")

    if compare:
        selected_telemetry2 = select_file(telemetry_files, "telemetry")
        metrics2, df2 = analyze_telemetry(selected_telemetry2, save_plots)
        file2_name = selected_telemetry2.stem

        # Comparison
        print("\n--- Comparison ---")
        comparison_df = pd.DataFrame({file1_name: metrics1, file2_name: metrics2})
        print(comparison_df)

        # Plot comparisons
        output_folder = selected_telemetry1.parent
        groups = {
            "RTT": [m for m in comparison_df.index if "RTT" in m],
            "Current": [m for m in comparison_df.index if "Current" in m],
            "Power": [m for m in comparison_df.index if "Power" in m or "Energy" in m],
            "Time": [m for m in comparison_df.index if "Time" in m],
            "Events": [m for m in comparison_df.index if m in ["Detachments", "Reattachments", "Parent Switches"]],
            "Percentage": [m for m in comparison_df.index if "(%)" in m],
        }
        for group, metrics_list in groups.items():
            if metrics_list:
                sub_df = comparison_df.loc[metrics_list].dropna(how='all')
                if not sub_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sub_df.T.plot(kind='bar', ax=ax)
                    ax.set_title(f"{group} Comparison")
                    ax.set_ylabel(group)
                    ax.legend(loc='best')
                    plt.xticks(rotation=0)
                    if save_plots:
                        out_path = output_folder / f"comparison_{group.lower()}.png"
                        fig.savefig(out_path, bbox_inches="tight", dpi=300)
                        print(f"Saved: {out_path}")

        # Time series comparison line graphs
        common_cols = set(df1.columns) & set(df2.columns)
        plot_cols = ["rtt_ms", "lqi_in", "lqi_out", "age_s", "tx_total", "rx_total",
                     "tx_err_cca", "tx_retry", "rx_err_fcs", "avg_current_uA"]
        for col in plot_cols:
            if col in common_cols and df1[col].notna().any() and df2[col].notna().any():
                fig, ax = plt.subplots(figsize=(12, 6))
                # Relative time for df1
                if not df1.empty:
                    rel_time1 = (df1["timestamp"] - df1["timestamp"].min()).dt.total_seconds()
                    ax.plot(rel_time1, df1[col], label=file1_name)
                # Relative time for df2
                if not df2.empty:
                    rel_time2 = (df2["timestamp"] - df2["timestamp"].min()).dt.total_seconds()
                    ax.plot(rel_time2, df2[col], label=file2_name)
                ax.set_title(f"{col} Comparison over Relative Time")
                ax.set_xlabel("Relative Time (seconds)")
                ax.set_ylabel(col)
                ax.legend()
                ax.grid(True)
                if save_plots:
                    out_path = output_folder / f"comparison_{col}.png"
                    fig.savefig(out_path, bbox_inches="tight", dpi=300)
                    print(f"Saved: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    main()