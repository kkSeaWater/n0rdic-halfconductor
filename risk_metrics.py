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

# ---------- Helpers ----------
def find_latest_csv(folder):
    files = glob.glob(os.path.join(folder, "correlated_risk_telemetry_*.csv"))
    if not files:
        files = glob.glob(os.path.join(folder, "risk_telemetry_*.csv"))
    if not files:
        raise FileNotFoundError(f"No (correlated_) risk telemetry CSV found in {folder}")
    return max(files, key=os.path.getmtime)

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
            r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]\s+event=predictive_switch\s+ok=(True|False)\s+elapsed=([\d\.]+)s', line)
        if match:
            time_str = match.group(1)
            ok = match.group(2) == 'True'
            elapsed = float(match.group(3))
            try:
                date_str = Path(txt_path).stem.split('_')[-2]
                datetime_str = f"{date_str} {time_str}"
                switch_time = datetime.strptime(datetime_str, '%Y%m%d %H:%M:%S.%f')
            except Exception:
                today = datetime.now().strftime('%Y%m%d')
                switch_time = datetime.strptime(f"{today} {time_str}", '%Y%m%d %H:%M:%S.%f')
            switches.append((switch_time, elapsed, ok))
    return switches

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
                           color=parent_color.get(current_parent, None), alpha=0.1,
                           label=label_text if current_parent not in ax.get_legend_handles_labels()[1] else "")
            start_time = row["timestamp"]
            current_parent = parent
    if current_parent is not None and start_time is not None:
        label_text = 'No Parent' if current_parent == 'none' else current_parent
        ax.axvspan(start_time, df["timestamp"].iloc[-1],
                   color=parent_color.get(current_parent, None), alpha=0.1,
                   label=label_text if current_parent not in ax.get_legend_handles_labels()[1] else "")

def mark_switch_durations(ax, switches):
    for switch_time, elapsed, ok in switches:
        end_switch = switch_time + timedelta(seconds=elapsed)
        ax.axvspan(switch_time, end_switch, alpha=0.3)
        mid = switch_time + timedelta(seconds=elapsed / 2)
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"{elapsed:.2f}s",
                horizontalalignment='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))

def savefig(fig, csv_path, suffix):
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

# ---------- NEW: Attachment Pie ----------
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

def plot_attachment_pie(durations, csv_path):
    if not durations:
        print("No durations to plot for attachment pie.")
        return None
    labels = list(durations.keys())
    values = [durations[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=labels, autopct=lambda p: f"{p:.1f}%")
    ax.set_title("Attachment Time Distribution")
    savefig(fig, csv_path, "attachment_pie")
    return fig

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=r"C:\Users\adire\Desktop\nordic_logs",
                        help="Folder with telemetry CSV")
    args = parser.parse_args()

    csv_path = find_latest_csv(args.folder)
    print(f"Analyzing: {csv_path}")
    txt_path = find_matching_txt(csv_path)
    switches = parse_switch_durations(txt_path)
    df = load_csv(csv_path)

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

    plt.show()

if __name__ == "__main__":
    main()


