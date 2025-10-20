
import os
import glob
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import io

# ---------- Helpers ----------
def find_latest_csv(folder):
    files = glob.glob(os.path.join(folder, "*telemetry_*.csv"))  # Catch both child_ and correlated_
    if not files:
        raise FileNotFoundError(f"No telemetry CSV found in {folder}")
    return max(files, key=os.path.getmtime)

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

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Analyze OpenThread telemetry CSV.")
    parser.add_argument("--folder", default=r"C:\Users\adire\Desktop\nordic_logs", help="Directory containing telemetry CSVs")
    args = parser.parse_args()
    
    csv_path = find_latest_csv(args.folder)
    print(f"\nUsing telemetry file: {csv_path}\n")
    df = load_csv(csv_path)
    df["parent_rloc16"] = df["parent_rloc16"].fillna("none")  # Fill for detached/none
    df.loc[df['state'] != 'child', 'parent_rloc16'] = 'none'  # Set to none when not in child state
    t = df["timestamp"]
    
    has_power = "avg_current_uA" in df.columns
    
    # Helper to create plot with common elements
    def create_plot(title, ylabel):
        fig, ax = plt.subplots()
        mark_detached(ax, df)
        add_event_markers(ax, df)
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        return fig, ax
    
    # 1) RTT
    if df["rtt_ms"].notna().any():
        fig1, ax1 = create_plot("RTT over time", "RTT (ms)")
        ax1.plot(t, df["rtt_ms"], color="tab:blue", label="RTT (ms)")
        # Add moving average for smoother trend
        if len(df) > 5:
            ma = df["rtt_ms"].rolling(window=5).mean()
            ax1.plot(t, ma, color="tab:orange", label="5-pt MA")
        ax1.legend()
        savefig(fig1, csv_path, "rtt")
    else:
        warnings.warn("Skipping RTT plot: No valid data.")
    
    # 2) LQI
    if df["lqi_in"].notna().any() or df["lqi_out"].notna().any():
        fig2, ax2 = create_plot("LQI over time", "LQI")
        if "lqi_in" in df: ax2.plot(t, df["lqi_in"], label="LQI in", color="tab:orange")
        if "lqi_out" in df: ax2.plot(t, df["lqi_out"], label="LQI out", color="tab:green")
        ax2.legend()
        savefig(fig2, csv_path, "lqi")
    else:
        warnings.warn("Skipping LQI plot: No valid data.")
    
    # 3) MAC deltas
    def delta(col):
        return df[col].diff().fillna(0) if col in df else None
    
    d_retry = delta("tx_retry")
    d_cca = delta("tx_err_cca")
    d_fcs = delta("rx_err_fcs")
    if any(d is not None for d in [d_retry, d_cca, d_fcs]):
        fig3, ax3 = create_plot("MAC error deltas per tick", "Count")
        if d_retry is not None: ax3.plot(t, d_retry, label="Δ retry")
        if d_cca is not None: ax3.plot(t, d_cca, label="Δ CCA")
        if d_fcs is not None: ax3.plot(t, d_fcs, label="Δ FCS")
        ax3.legend()
        savefig(fig3, csv_path, "mac_deltas")
    else:
        warnings.warn("Skipping MAC deltas plot: No valid data.")
    
    # 4) State timeline
    state_map = {"disabled": 0, "detached": 1, "child": 2, "router": 3, "leader": 4, "": -1}
    y = df["state"].map(lambda s: state_map.get(str(s).lower(), -1))
    fig4, ax4 = create_plot("Node state timeline", "State")
    ax4.step(t, y, where="post", color="tab:purple")
    ax4.set_yticks(list(state_map.values()))
    ax4.set_yticklabels(list(state_map.keys()))
    savefig(fig4, csv_path, "state")
    
    # 4.5) Parent timeline (new graph)
    if "parent_rloc16" in df.columns and df["parent_rloc16"].notna().any():
        unique_parents = sorted(df["parent_rloc16"].unique())
        parent_map = {p: i for i, p in enumerate(unique_parents)}
        y_parent = df["parent_rloc16"].map(parent_map)
        fig_parent, ax_parent = create_plot("Parent Attachment over Time", "Parent RLOC16")
        ax_parent.step(t, y_parent, where="post", color="tab:orange", linewidth=2)
        ax_parent.set_yticks(list(parent_map.values()))
        ax_parent.set_yticklabels(list(parent_map.keys()))
        # Annotate changes
        changes = df["parent_rloc16"].ne(df["parent_rloc16"].shift()).index[df["parent_rloc16"].ne(df["parent_rloc16"].shift())]
        for idx in changes:
            if idx > 0:
                ax_parent.annotate(df.loc[idx, "parent_rloc16"], (t[idx], y_parent[idx]), xytext=(5, 5), textcoords='offset points', fontsize=8)
        savefig(fig_parent, csv_path, "parent")
    else:
        warnings.warn("Skipping parent plot: No valid data.")
    
    # 5) Power (if available)
    if has_power and df["avg_current_uA"].notna().any():
        fig5, ax5 = create_plot("Average Current over time", "Current (μA)")
        ax5.plot(t, df["avg_current_uA"], color="tab:red", label="Avg Current (μA)", linewidth=2)
        # Tight y-limits to show variations better
        i_min, i_max = df["avg_current_uA"].min(), df["avg_current_uA"].max()
        ax5.set_ylim(i_min - 20, i_max + 20)
        ax5.legend()
        savefig(fig5, csv_path, "power")
    elif has_power:
        warnings.warn("Skipping power plot: No valid data.")
    
    # 6) Transmission and Power Spikes (original)
    if has_power and df["avg_current_uA"].notna().any() and all(col in df for col in ["tx_total", "rx_total"]):
        df['tx_delta'] = df['tx_total'].diff().fillna(0)
        df['rx_delta'] = df['rx_total'].diff().fillna(0)
        df['msg_activity'] = df['tx_delta'] + df['rx_delta']
        df['current_delta'] = df['avg_current_uA'].diff().fillna(0)  # Now using delta for spikes
        
        fig6, ax1 = plt.subplots(figsize=(12, 6))
        mark_detached(ax1, df)
        add_event_markers(ax1, df)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Power delta on left axis for spike visibility (instead of absolute, which is flat)
        ax1.plot(t, df["current_delta"], color="tab:red", label="Δ Current (μA)", linewidth=2)
        ax1.set_title("Message Transmission Spikes and Power Spikes")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Δ Current (μA)", color="tab:red")
        ax1.tick_params(axis='y', labelcolor="tab:red")
        # Tight limits
        d_min, d_max = df["current_delta"].min(), df["current_delta"].max()
        ax1.set_ylim(d_min - 50, d_max + 50)  # Amplified for visibility
        
        # Twin axis for message activity spikes
        ax2 = ax1.twinx()
        # Compute bar width based on average time delta (in days for matplotlib)
        delta_t_days = (t - t.shift(1)).mean().total_seconds() / 86400 if len(t) > 1 else 0.001
        ax2.bar(t, df["msg_activity"], width=delta_t_days * 0.8, color="tab:blue", alpha=0.6, label="Msg Activity (Δ TX+RX)")
        ax2.set_ylabel("Message Activity (packets)", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor="tab:blue")
        
        fig6.legend(loc="upper right")
        savefig(fig6, csv_path, "transmission_power_spikes")
    else:
        warnings.warn("Skipping transmission/power spikes plot: Missing required columns (avg_current_uA, tx_total, rx_total).")
    
    # 7) Transmission and Power Spikes with Parent Periods (new)
    if has_power and df["avg_current_uA"].notna().any() and all(col in df for col in ["tx_total", "rx_total"]) and "parent_rloc16" in df.columns:
        fig7, ax3 = plt.subplots(figsize=(12, 6))
        mark_detached(ax3, df)
        add_event_markers(ax3, df)
        mark_parent_periods(ax3, df)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        # Power delta
        ax3.plot(t, df["current_delta"], color="tab:red", label="Δ Current (μA)", linewidth=2)
        ax3.set_title("Message Transmission Spikes and Power Spikes with Parent Periods")
        ax3.set_xlabel("Timestamp")
        ax3.set_ylabel("Δ Current (μA)", color="tab:red")
        ax3.tick_params(axis='y', labelcolor="tab:red")
        ax3.set_ylim(d_min - 50, d_max + 50)
        
        # Twin axis for message activity
        ax4 = ax3.twinx()
        ax4.bar(t, df["msg_activity"], width=delta_t_days * 0.8, color="tab:blue", alpha=0.6, label="Msg Activity (Δ TX+RX)")
        ax4.set_ylabel("Message Activity (packets)", color="tab:blue")
        ax4.tick_params(axis='y', labelcolor="tab:blue")
        
        fig7.legend(loc="upper right")
        savefig(fig7, csv_path, "transmission_power_spikes_with_parents")
    else:
        warnings.warn("Skipping transmission/power spikes with parents plot: Missing required columns.")
    
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