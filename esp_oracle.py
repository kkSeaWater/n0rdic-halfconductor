"""
esp_oracle.py

Oracle-style "Transmission and Power with Parent Periods" plot for ESP tests.

Files (same timestamp token, e.g. 20251120_144157):

    esp_telemetry_YYYYMMDD_HHMMSS.csv
    esp_parent_events_YYYYMMDD_HHMMSS.csv
    esp_log_YYYYMMDD_HHMMSS.txt   (optional, for BetterParent markers)

Behaviour
---------
- By default: uses the **latest** telemetry file and its matching parent_events/log
  (based on the timestamp part of the filename).

- Override with:
    python esp_oracle.py --stamp 20251120_144157
  or:
    python esp_oracle.py --stamp 144157

Parent semantics
----------------
- Start parent = from_rloc16 of the FIRST parent_switch event.
- At each parent_switch: parent -> to_rloc16.
- Telemetry parent_rloc16 is NOT used for changes.
- effective_parent:
    * 'No Parent' if state is blank/detached or parent empty
    * otherwise = parent id string

BetterParent markers
--------------------
- Parse the matching esp_log_*.txt.
- For each line containing "Attach attempt" AND "BetterParent":
    draw a grey dotted vertical line at that timestamp.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ========= generic helpers ========= #

def parse_timestamp(series):
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    return ts.dt.tz_convert(None)


def has_data(s):
    return isinstance(s, pd.Series) and not s.empty and s.notna().any()


def parse_args():
    parser = argparse.ArgumentParser(
        description="ESP oracle-style Transmission & Power plot",
        add_help=True,
    )
    parser.add_argument(
        "--stamp",
        type=str,
        default=None,
        help=("Timestamp id to select a specific run, e.g. "
              "'20251120_144157' or just '144157'. "
              "If omitted, the latest run is used."),
    )
    # Spyder passes extra stuff like --wdir; ignore them
    args, _ = parser.parse_known_args()
    return args


# ========= file selection ========= #

def _extract_stamp_from_telemetry_path(p: Path) -> str:
    """
    esp_telemetry_20251120_144157.csv -> '20251120_144157'
    """
    stem = p.stem  # 'esp_telemetry_20251120_144157'
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2]
    return stem


def select_esp_pair(stamp):
    """
    Choose telemetry + parent_events files and return:
        (telemetry_path, parent_events_path, token)

    - If stamp is None:
        pick the telemetry file with the **largest** timestamp token.
    - If stamp is provided:
        match any telemetry whose token == stamp OR token.endswith(stamp).
    """
    tele_files = sorted(Path(".").glob("esp_telemetry_*.csv"))
    if not tele_files:
        raise FileNotFoundError("No esp_telemetry_*.csv files found in this folder.")

    if stamp is not None:
        stamp = stamp.strip()
        candidates = []
        for f in tele_files:
            token = _extract_stamp_from_telemetry_path(f)
            if token == stamp or token.endswith(stamp):
                candidates.append((token, f))
        if not candidates:
            raise FileNotFoundError(
                f"No esp_telemetry_*.csv matches stamp '{stamp}'. "
                "Available tokens: "
                + ", ".join(_extract_stamp_from_telemetry_path(p) for p in tele_files)
            )
        token, tel_path = sorted(candidates, key=lambda x: x[0])[-1]
    else:
        tele_files_sorted = sorted(tele_files, key=_extract_stamp_from_telemetry_path)
        tel_path = tele_files_sorted[-1]
        token = _extract_stamp_from_telemetry_path(tel_path)

    parent_name = f"esp_parent_events_{token}.csv"
    parent_path = Path(parent_name)
    if not parent_path.exists():
        raise FileNotFoundError(
            f"Expected matching parent events file '{parent_name}' not found."
        )

    print("Telemetry file   :", tel_path.name)
    print("Parent events file:", parent_path.name)
    return tel_path, parent_path, token


# ========= load data ========= #

def load_parent_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # columns: timestamp_iso, event_type, from_rloc16, to_rloc16, raw_line
    df["timestamp"] = parse_timestamp(df["timestamp_iso"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns include:
    # timestamp_iso, state, parent_rloc16, parent_extaddr,
    # lqi_in, lqi_out, age_s, tx_total, rx_total, tx_err_cca, tx_retry, rx_err_fcs, ...
    df["timestamp"] = parse_timestamp(df["timestamp_iso"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for c in ["tx_total", "rx_total", "avg_current_uA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_betterparent_times(log_path: Path):
    """
    Parse the ESP log and return a list of timestamps where we see
    'Attach attempt' and 'BetterParent' on the same line.
    """
    times = []
    if not log_path.exists():
        print(f"[INFO] No log file {log_path.name} found; skipping BetterParent markers.")
        return times

    print("Log file         :", log_path.name)

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Attach attempt" in line and "BetterParent" in line:
                # expected format:
                # 2025-11-20T13:41:58.244455+00:00 I(539469) OPENTHREAD...
                ts_str = line.split(" ", 1)[0]
                try:
                    ts = pd.to_datetime(ts_str, utc=True)
                    ts = ts.tz_convert(None)
                    times.append(ts)
                except Exception:
                    continue

    return times


# ========= parent timeline from *events only* ========= #

def build_parent_from_events(df_tel: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Semantics:

    - Start parent = from_rloc16 of the FIRST parent_switch.
    - At each parent_switch, switch to to_rloc16.
    - Telemetry parent_rloc16 is ignored for changes.

    effective_parent:
      - 'No Parent' if state blank/detached or parent empty
      - otherwise the parent id.
    """
    df = df_tel.copy()

    switches = df_events[df_events["event_type"] == "parent_switch"].copy()
    switches = switches.sort_values("timestamp").reset_index(drop=True)

    times = df["timestamp"]
    parents_from_events = []

    if switches.empty:
        parent_current = ""
        idx_sw = 0
        n_sw = 0
    else:
        first = switches.iloc[0]
        init = first.get("from_rloc16")
        if pd.isna(init) or str(init).strip().lower() in ("", "nan", "none"):
            init = first.get("to_rloc16")
        parent_current = "" if pd.isna(init) else str(init).strip()
        idx_sw = 0
        n_sw = len(switches)

    for t in times:
        while idx_sw < n_sw and t >= switches.loc[idx_sw, "timestamp"]:
            new_p = switches.loc[idx_sw, "to_rloc16"]
            parent_current = "" if pd.isna(new_p) else str(new_p).strip()
            idx_sw += 1
        parents_from_events.append(parent_current)

    df["parent_from_events"] = parents_from_events

    def is_blank_state(v):
        if pd.isna(v):
            return True
        s = str(v).strip().lower()
        return s == "" or s == "blank"

    effective = []
    for _, row in df.iterrows():
        st = row.get("state", "")
        base = str(row.get("parent_from_events", "")).strip()
        if is_blank_state(st) or str(st).strip().lower() == "detached":
            effective.append("No Parent")
        elif base == "" or base.lower() in ("nan", "none"):
            effective.append("No Parent")
        else:
            effective.append(base)

    df["effective_parent"] = effective
    return df


# ========= backgrounds & markers ========= #

def mark_parent_periods(ax, df, alpha=0.35, zorder=0.02):
    """
    Shade background based on df['effective_parent'].
    One colour per parent, pink for 'No Parent'.
    """
    if df.empty or "timestamp" not in df.columns or "effective_parent" not in df.columns:
        return

    eff = df["effective_parent"]
    uniq = list(pd.unique(eff))

    no_parent_color = "#f8c8cf"
    real_parents = [u for u in uniq if u != "No Parent"]
    base = plt.get_cmap("tab20", max(len(real_parents), 1))

    color_map = {"No Parent": no_parent_color}
    for idx, p in enumerate(real_parents):
        color_map[p] = base(idx)

    cur = None
    start = None
    for i, row in df.iterrows():
        cur_eff = eff.loc[i]
        if cur_eff != cur:
            if cur is not None and start is not None:
                label = cur
                _, labels = ax.get_legend_handles_labels()
                ax.axvspan(
                    start,
                    row["timestamp"],
                    color=color_map.get(cur, "#dddddd"),
                    alpha=alpha,
                    zorder=zorder,
                    label=label if label not in labels else "",
                )
            start = row["timestamp"]
            cur = cur_eff

    if cur is not None and start is not None:
        label = cur
        _, labels = ax.get_legend_handles_labels()
        ax.axvspan(
            start,
            df["timestamp"].iloc[-1],
            color=color_map.get(cur, "#dddddd"),
            alpha=alpha,
            zorder=zorder,
            label=label if label not in labels else "",
        )


def add_parent_switch_markers(ax, df_events: pd.DataFrame):
    sw = df_events[df_events["event_type"] == "parent_switch"].copy()
    if sw.empty:
        return

    used_label = False
    y_top = ax.get_ylim()[1]
    for _, row in sw.iterrows():
        t = row["timestamp"]
        frm = str(row.get("from_rloc16", "")).strip()
        to = str(row.get("to_rloc16", "")).strip()
        lbl = "Parent Switch" if not used_label else ""
        ax.axvline(t, linestyle="--", color="black", alpha=1.0, label=lbl, linewidth=2.5)
        used_label = True
        if frm and to:
            ax.text(
                t,
                y_top * 0.985,
                f"{frm}->{to}",
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                color="black",
                bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
            )


def add_betterparent_markers(ax, times):
    """
    Grey dotted lines at each 'Attach attempt, BetterParent' timestamp.
    """
    if not times:
        return

    used_label = False
    for ts in times:
        lbl = "Attach attempt (BetterParent)" if not used_label else ""
        ax.axvline(ts, linestyle=":", color="green", alpha=1.0, label=lbl, linewidth=2.5)
        used_label = True


def setup_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# ========= main plot ========= #

def plot_transmission_and_power(df_tel: pd.DataFrame,
                                df_events: pd.DataFrame,
                                betterparent_times):
    t = df_tel["timestamp"]
    tx_total = pd.to_numeric(df_tel["tx_total"], errors="coerce")
    rx_total = pd.to_numeric(df_tel["rx_total"], errors="coerce")
    i = pd.to_numeric(df_tel["avg_current_uA"], errors="coerce") if "avg_current_uA" in df_tel.columns else None

    delta_tx = tx_total.diff().fillna(0)
    delta_rx = rx_total.diff().fillna(0)
    msg_activity = delta_tx + delta_rx

    interval_s = t.diff().dt.total_seconds().median()
    if np.isnan(interval_s) or interval_s <= 0:
        bar_width = (1 / 86400) * 0.8
    else:
        bar_width = (interval_s / 86400) * 0.8

    fig, ax_left = plt.subplots(figsize=(12, 5))

    mark_parent_periods(ax_left, df_tel, alpha=0.35, zorder=0.02)

    # Left axis: current (if available)
    if i is not None and has_data(i):
        ax_left.plot(t, i, color="tab:red", label="Avg Current (μA)")
    ax_left.set_ylabel("Current (μA)", color="tab:red")
    ax_left.tick_params(axis="y", labelcolor="tab:red")
    ax_left.grid(True)

    # Right axis: message activity
    ax_right = ax_left.twinx()
    ax_right.bar(
        t,
        msg_activity,
        width=bar_width,
        align="center",
        color="tab:blue",
        alpha=0.7,
        label="Msg Activity (Δ TX+RX)",
    )
    ax_right.set_ylabel("Message Activity (packets)", color="tab:blue")
    ax_right.tick_params(axis="y", labelcolor="tab:blue")

    # Markers
    add_parent_switch_markers(ax_left, df_events)
    add_betterparent_markers(ax_left, betterparent_times)

    fig.suptitle("Transmission and Power with Parent Periods")
    setup_time_axis(ax_left)

    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plt.show()


# ========= main ========= #

def main():
    args = parse_args()
    tel_path, ev_path, token = select_esp_pair(args.stamp)

    df_tel = load_telemetry(tel_path)
    df_events = load_parent_events(ev_path)
    df_tel = build_parent_from_events(df_tel, df_events)

    log_path = Path(f"esp_log_{token}.txt")
    betterparent_times = load_betterparent_times(log_path)

    plot_transmission_and_power(df_tel, df_events, betterparent_times)


if __name__ == "__main__":
    main()
