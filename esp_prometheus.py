"""
esp_oracle.py

Oracle-style "Transmission and Power with Parent Periods" plot for ESP tests.

Files (same timestamp token, e.g. 20251120_144157):

    esp_telemetry_YYYYMMDD_HHMMSS.csv
    esp_parent_events_YYYYMMDD_HHMMSS.csv
    esp_log_YYYYMMDD_HHMMSS.txt   (optional, for BetterParent markers)

Optional PPK2 spreadsheet:

    - New preferred: absolute timestamp + Current(uA)
    - Legacy:        Timestamp(ms) + Current(uA) (relative ms)

Behaviour:

    - If you call with --stamp:
        * Uses esp_telemetry_<stamp>.csv and esp_parent_events_<stamp>.csv
          where <stamp> matches the token, or the end of the token.

    - If you call with no args:
        * Interactive chooser: picks the newest telemetry/events pair and
          optionally a PPK2 CSV.

Main output:

    A single big figure "Transmission and Power with Parent Periods" that shows:

      - Shaded background regions per "effective parent" over time
        (including "No Parent").
      - Left Y-axis: current (μA) from:
          * Telemetry 'avg_current_uA'
          * Row-wise PPK averages (if aligned)
          * Optional raw PPK trace
      - Right Y-axis: message activity (Δtx_total + Δrx_total) as bars.
      - Vertical lines for:
          * parent_switch events (with text labels "old->new")
          * log-derived parent-search-related events (BetterParent, Attach, etc.)
      - Optional third Y-axis: RSS (dBm) derived from "rss:" in ESP log.
      - Title annotated with last known RSS from the log, if present.
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


# ========= generic helpers ========= #

def parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Parse timestamps which might be:

      - ISO-like, possibly with 'Z':
            2025-11-20T14:41:57.123Z
            2025-11-20T14:41:57

      - Or plain HH:MM:SS(.sss)
        In this case we assume "today" just so matplotlib can plot it.
    """
    s = series.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", utc=True)

    mask_plain = out.isna() & s.str.match(r"^\d\d:\d\d:\d\d(\.\d+)?$")
    if mask_plain.any():
        today = pd.Timestamp.now(tz="UTC").normalize()
        tmp = "1970-01-01 " + s[mask_plain]
        out.loc[mask_plain] = pd.to_datetime(tmp, errors="coerce", utc=True)

    return out.dt.tz_convert(None)


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
              "'20251120_144157' or just '144157' "
              "(matches suffix of the telemetry filename)."),
    )
    parser.add_argument(
        "--ppk",
        type=str,
        default=None,
        help="Optional path to PPK2 CSV (absolute or relative)",
    )
    return parser.parse_args()


# ========= file selection ========= #

def extract_token_from_name(path: Path) -> str:
    """
    From a filename like 'esp_telemetry_20251120_144157.csv'
    return '20251120_144157'.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[-2:])
    return stem


def select_esp_pair(stamp: str | None):
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
        candidates = []
        for f in tele_files:
            token = extract_token_from_name(f)
            if token == stamp or token.endswith(stamp):
                candidates.append((token, f))
        if not candidates:
            raise FileNotFoundError(
                f"No telemetry file whose token == '{stamp}' "
                "or token.endswith(stamp) was found."
            )
    else:
        candidates = [(extract_token_from_name(f), f) for f in tele_files]

    candidates.sort(key=lambda x: x[0])
    token, tel_path = candidates[-1]

    parent_path = Path(f"esp_parent_events_{token}.csv")
    if not parent_path.exists():
        raise FileNotFoundError(f"Matching parent events file {parent_path} not found.")

    print("Telemetry file   :", tel_path.name)
    print("Parent events    :", parent_path.name)
    return tel_path, parent_path, token


def interactive_select_files():
    """
    Interactive version of select_esp_pair() that also lets you
    optionally choose a PPK2 CSV from the current directory.

    Returns:
        tel_path, parent_path, token, ppk_path_or_None
    """
    tele_files = sorted(Path(".").glob("esp_telemetry_*.csv"))
    if not tele_files:
        raise FileNotFoundError("No esp_telemetry_*.csv files found in this folder.")

    # Choose the newest token
    candidates = [(extract_token_from_name(f), f) for f in tele_files]
    candidates.sort(key=lambda x: x[0])
    token, tel_path = candidates[-1]

    parent_path = Path(f"esp_parent_events_{token}.csv")
    if not parent_path.exists():
        raise FileNotFoundError(f"Matching parent events file {parent_path} not found.")

    print("Telemetry file   :", tel_path.name)
    print("Parent events    :", parent_path.name)

    # Look for possible PPK CSVs
    ppk_candidates = sorted(Path(".").glob("ppk_*.csv"))
    ppk_path = None
    if ppk_candidates:
        print("\nAvailable PPK CSV files:")
        for i, p in enumerate(ppk_candidates, start=1):
            print(f"  {i}) {p.name}")
        choice = input("Select PPK file number (or Enter for none): ").strip()
        if choice:
            idx = int(choice)
            if 1 <= idx <= len(ppk_candidates):
                ppk_path = ppk_candidates[idx - 1]
                print("PPK file         :", ppk_path.name)

    return tel_path, parent_path, token, ppk_path


# ========= load data ========= #

def load_noack_events(log_path: Path) -> pd.DataFrame:
    """
    Parse MAC NoAck failures from the ESP log.

    We look for lines like:

        Mac-----------: Frame tx attempt 7/16 failed, error:NoAck, len:40, ...
                                                   src:0x7003, dst:0x7000, ...

    and build a DataFrame with:
        timestamp, attempt, max_attempts, length, src, dst
    """
    if not log_path.exists():
        return pd.DataFrame()

    rows = []
    pattern = re.compile(
        r"Frame tx attempt (\d+)/(\d+) failed, error:NoAck, len:(\d+).*?"
        r"src:(0x[0-9a-fA-F]+), dst:(0x[0-9a-fA-F]+)"
    )

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            # First token is the ISO-like timestamp.
            ts_str = line.split(" ", 1)[0].strip()
            ts = parse_timestamp(pd.Series([ts_str]))[0]

            if pd.isna(ts):
                continue

            rows.append(
                {
                    "timestamp": ts,
                    "attempt": int(m.group(1)),
                    "max_attempts": int(m.group(2)),
                    "length": int(m.group(3)),
                    "src": m.group(4),
                    "dst": m.group(5),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_telemetry(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = parse_timestamp(df["timestamp_iso"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for c in ["tx_total", "rx_total", "avg_current_uA"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def load_parent_events(path: Path) -> pd.DataFrame:
    """
    Load the esp_parent_events_*.csv file and add a parsed timestamp column.

    Expected columns include at least:
        - timestamp_iso  (ISO-like string)
        - event_type     ("parent_switch", etc.)
        - from_rloc16    (may be empty / NaN)
        - to_rloc16

    Returns a DataFrame sorted by timestamp with a new 'timestamp' column.
    """
    df = pd.read_csv(path)
    df["timestamp"] = parse_timestamp(df["timestamp_iso"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def load_ppk_csv(ppk_path: Path) -> pd.DataFrame:
    """
    Load a PPK spreadsheet.

    New preferred format:
        timestamp column containing absolute timestamps (ISO-ish)
        and a current column with something like 'Current(uA)' in the name.

    Legacy format:
        a 'Timestamp(ms)' column (relative) and 'Current(uA)'.

    This function auto-detects which style is present and returns a
    uniform DataFrame with:

        timestamp : datetime64[ns]
        current_uA: float
    """
    df_raw = pd.read_csv(ppk_path)

    # 1) Absolute timestamps?
    ts_abs_col = None
    cur_col = None
    for c in df_raw.columns:
        lc = c.lower()
        if "current" in lc and "ua" in lc:
            cur_col = c
        if ("timestamp" in lc or "time" in lc) and "ms" not in lc:
            ts_abs_col = c

    if ts_abs_col is not None and cur_col is not None:
        ts_abs = pd.to_datetime(df_raw[ts_abs_col], errors="coerce", utc=True)
        name = getattr(ts_abs, "tzinfo", None)
        if name is not None:
            ts_abs = ts_abs.dt.tz_convert(None)
        else:
            ts_abs = ts_abs.dt.tz_localize(None)

        df = pd.DataFrame(
            {
                "timestamp": ts_abs,
                "current_uA": pd.to_numeric(df_raw[cur_col], errors="coerce"),
            }
        )
        df = df.dropna(subset=["timestamp", "current_uA"]).reset_index(drop=True)
        return df

    # 2) Legacy / fallback: look for a relative 'Timestamp(ms)' style column
    ts_rel_col = None
    cur_col = None
    for c in df_raw.columns:
        lc = c.lower()
        if "current" in lc and "ua" in lc:
            cur_col = c
        if "timestamp" in lc and "ms" in lc:
            ts_rel_col = c

    if ts_rel_col is None or cur_col is None:
        raise ValueError("Could not detect suitable timestamp/current columns in PPK CSV.")

    t_rel_ms = pd.to_numeric(df_raw[ts_rel_col], errors="coerce")
    base = pd.Timestamp.now().normalize()
    ts_abs = base + pd.to_timedelta(t_rel_ms, unit="ms")
    df = pd.DataFrame(
        {
            "timestamp": ts_abs,
            "current_uA": pd.to_numeric(df_raw[cur_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["timestamp", "current_uA"]).reset_index(drop=True)
    return df


def align_ppk_to_telemetry(df_ppk: pd.DataFrame, df_tel: pd.DataFrame) -> pd.DataFrame:
    """
    Trim PPK samples so they fall within telemetry time span.
    """
    if df_ppk.empty or df_tel.empty:
        return df_ppk

    t0 = df_tel["timestamp"].iloc[0]
    t1 = df_tel["timestamp"].iloc[-1]
    df = df_ppk[(df_ppk["timestamp"] >= t0) & (df_ppk["timestamp"] <= t1)].copy()
    df = df.reset_index(drop=True)
    return df


def attach_ppk_averages_to_telemetry(df_tel: pd.DataFrame, df_ppk: pd.DataFrame) -> pd.DataFrame:
    """
    For each telemetry interval [t_i, t_{i+1}), compute the mean PPK current_uA
    of samples whose timestamps fall inside that interval and store it in
    a new column 'ppk_avg_current_uA'.

    The last telemetry row uses [t_{last}, t_{last} + median_delta].
    """
    if df_tel.empty or df_ppk.empty:
        df_tel["ppk_avg_current_uA"] = np.nan
        return df_tel

    times = df_tel["timestamp"].values
    ppk_ts = df_ppk["timestamp"].values
    ppk_i = df_ppk["current_uA"].values

    dt = df_tel["timestamp"].diff().dt.total_seconds().median()
    if np.isnan(dt) or dt <= 0:
        dt = 1.0

    bucket_means = np.full(len(times), np.nan, dtype=float)

    j = 0
    n_ppk = len(ppk_ts)
    for i in range(len(times)):
        start = times[i]
        if i < len(times) - 1:
            end = times[i + 1]
        else:
            end = start + pd.Timedelta(seconds=dt)

        # Skip PPK samples before this telemetry interval
        while j < n_ppk and ppk_ts[j] < start:
            j += 1

        k = j
        values = []
        # Collect samples inside the interval
        while k < n_ppk and ppk_ts[k] < end:
            values.append(ppk_i[k])
            k += 1

        if values:
            bucket_means[i] = float(np.mean(values))

        j = k  # next interval can start from here

    df_tel = df_tel.copy()
    df_tel["ppk_avg_current_uA"] = bucket_means
    return df_tel

def load_betterparent_times(log_path: Path):
    """Parse the ESP log for periodic parent-search check events only.

    We *only* mark the moments when the Periodic Parent Search "check
    interval" has passed, i.e. when OpenThread logs:

        Mle-----------: PeriodicParentSearch: Check interval passed

    That gives us one dotted line per "parent-search tick" (roughly once per
    configured interval, e.g. ~60 s), instead of spamming for every MLE log.
    """
    times: list[pd.Timestamp] = []
    if not log_path.exists():
        print(f"[INFO] No log file {log_path.name} found; skipping parent-search markers.")
        return times

    print("Log file         :", log_path.name)

    target_substring = "PeriodicParentSearch: Check interval passed"

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if target_substring not in line:
                continue

            # First token should be the ISO timestamp:
            # 2025-11-26T13:52:33.772440+00:00 ...
            ts_str = line.split(" ", 1)[0].strip()
            try:
                ts = parse_timestamp(pd.Series([ts_str]))[0]
                if pd.notna(ts):
                    times.append(ts)
            except Exception:
                continue

    return times



def load_rss_series(log_path: Path):
    """Extract an RSS (dBm) time-series from the ESP log.

    We look for any line containing ``rss:<value>`` (for example the
    MeshForwarder "Received IPv6 ... rss:-75.0" lines) and build a small
    DataFrame with:

        timestamp : parsed from the ISO-like token at the start of the line
        rss       : float, in dBm

    Returns
    -------
    df_rss : pd.DataFrame | None
        DataFrame with columns ["timestamp", "rss"], or None if no RSS
        values were found.
    last_rss : tuple[pd.Timestamp, float] | None
        (timestamp, rss) of the last RSS sample in the log, or None.
    """
    if not log_path.exists():
        return None, None

    ts_strings: list[str] = []
    rss_values: list[float] = []

    pattern = re.compile(r"rss:(-?\d+(?:\.\d+)?)")

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            ts_str = line.split(" ", 1)[0].strip()
            ts_strings.append(ts_str)
            try:
                rss_values.append(float(m.group(1)))
            except ValueError:
                rss_values.append(np.nan)

    if not ts_strings:
        return None, None

    ts_series = parse_timestamp(pd.Series(ts_strings))
    df_rss = pd.DataFrame({"timestamp": ts_series, "rss": rss_values})
    df_rss = df_rss.dropna(subset=["timestamp"]).reset_index(drop=True)

    last_rss: tuple[pd.Timestamp, float] | None = None
    if not df_rss.empty:
        valid = df_rss.dropna(subset=["rss"])
        if not valid.empty:
            last_row = valid.iloc[-1]
            last_rss = (last_row["timestamp"], float(last_row["rss"]))

    return df_rss, last_rss


# ========= parent timeline from *events only* ========= #

def build_parent_from_events(df_tel: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
    """
    Build an 'effective_parent' timeline.

    Rules:

      - Baseline parent comes from telemetry (parent_rloc16/parent/parent_id).
      - If we have explicit parent_switch events, they override the baseline
        from the moment they occur.
      - 'No Parent' is *only* used when the node is detached / blank state,
        not just because we don't know the parent's RLOC.

    Resulting columns:
      - parent_from_events : parent id after combining telemetry + events
      - effective_parent   : what we actually shade on the plot
    """
    df = df_tel.copy()

    # 1) Choose a parent column from telemetry (if any)
    parent_col = None
    for cand in ("parent_rloc16", "parent", "parent_id"):
        if cand in df.columns:
            parent_col = cand
            break

    if parent_col is not None:
        def norm_parent(v):
            if pd.isna(v):
                return ""
            s = str(v).strip()
            return "" if s.lower() in ("", "nan", "none") else s

        parents_from_telemetry = df[parent_col].map(norm_parent)
    else:
        # No explicit parent column in telemetry – treat as unknown
        parents_from_telemetry = pd.Series([""] * len(df), index=df.index)

    # 2) Optional overrides from parent_switch events
    if df_events is not None and not df_events.empty:
        switches = df_events[df_events["event_type"] == "parent_switch"].copy()
        switches = switches.sort_values("timestamp").reset_index(drop=True)
    else:
        switches = pd.DataFrame()

    parents_from_events: list[str] = []
    parent_current: str | None = None
    idx_sw = 0
    n_sw = len(switches)

    for i, t in enumerate(df["timestamp"]):
        # Apply all switches up to this time
        while idx_sw < n_sw and switches["timestamp"].iloc[idx_sw] <= t:
            new_p = switches["to_rloc16"].iloc[idx_sw]
            parent_current = None if pd.isna(new_p) else str(new_p).strip()
            idx_sw += 1

        if parent_current is not None:
            parents_from_events.append(parent_current)
        else:
            # Fall back to telemetry's view of the parent
            parents_from_events.append(parents_from_telemetry.iloc[i])

    df["parent_from_events"] = parents_from_events

    # 3) Convert into effective_parent (attach vs no-parent)
    def is_no_parent_state(st):
        if pd.isna(st):
            return True
        s = str(st).strip().lower()
        # treat blank/detached as having no parent
        return s in ("", "blank", "detached")

    states = df["state"] if "state" in df.columns else pd.Series([""] * len(df))
    effective: list[str] = []

    for st, base in zip(states, df["parent_from_events"]):
        b = "" if pd.isna(base) else str(base).strip()
        if is_no_parent_state(st):
            # Only here do we claim "No Parent"
            effective.append("No Parent")
        elif b == "" or b.lower() in ("nan", "none"):
            # Attached but we don't know which parent
            effective.append("Attached (unknown parent)")
        else:
            # Attached and we know the parent RLOC
            effective.append(b)

    df["effective_parent"] = effective
    return df


    def is_blank_state(v):
        if pd.isna(v):
            return True
        s = str(v).strip().lower()
        return s == "" or s == "blank"

    effective: list[str] = []
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
    for i, p in enumerate(real_parents):
        color_map[p] = base(i)

    cur = None
    start = None

    y_min, y_max = ax.get_ylim()
    y_top = y_max if y_max > y_min else 1.0

    for _, row in df.iterrows():
        p = row["effective_parent"]
        t = row["timestamp"]
        if cur is None:
            cur = p
            start = t
            continue
        if p != cur:
            label = cur
            _, labels = ax.get_legend_handles_labels()
            ax.axvspan(
                start,
                t,
                color=color_map.get(cur, "#dddddd"),
                alpha=alpha,
                zorder=zorder,
                label=label if label not in labels else "",
            )
            cur = p
            start = t

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
    if df_events.empty:
        return

    switches = df_events[df_events["event_type"] == "parent_switch"].copy()
    if switches.empty:
        return

    y_min, y_max = ax.get_ylim()
    y_top = y_max if y_max > y_min else 1.0

    used_label = False
    for _, row in switches.iterrows():
        t = row["timestamp"]
        frm = str(row.get("from_rloc16", "")).strip()
        to = str(row.get("to_rloc16", "")).strip()
        lbl = "Parent switch (events CSV)" if not used_label else ""
        ax.axvline(t, color="black", linewidth=1.5, linestyle="-", label=lbl)
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
                bbox=dict(facecolor="white", alpha=0.8, linewidth=0),
            )


def add_betterparent_markers(ax, times):
    """Draw vertical markers for parent-search-related log events."""
    if not times:
        return

    used_label = False
    for ts in times:
        if pd.isna(ts):
            continue
        lbl = "Parent-search / MLE adv (log)" if not used_label else ""
        ax.axvline(ts, linestyle=":", linewidth=1.5, alpha=0.9, label=lbl)
        used_label = True
        
def add_noack_markers(ax, df_noack: pd.DataFrame | None):
    """
    Draw vertical markers for MAC NoAck failures.
    Each line = at least one failed transmission attempt with NoAck.
    """
    if df_noack is None or df_noack.empty:
        return

    used_label = False
    for _, row in df_noack.iterrows():
        t = row["timestamp"]
        if pd.isna(t):
            continue

        label = "MAC NoAck (tx failures)" if not used_label else ""
        ax.axvline(
            t,
            linestyle="--",   # dashed, distinct from parent-search dotted
            linewidth=1.0,
            alpha=0.6,
            label=label,
        )
        used_label = True



def setup_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# ========= main plot ========= #

def plot_transmission_and_power(
    df_tel: pd.DataFrame,
    df_events: pd.DataFrame,
    betterparent_times,
    df_noack: pd.DataFrame | None = None,
    df_ppk: pd.DataFrame | None = None,
    df_rss: pd.DataFrame | None = None,
    last_rss: tuple[pd.Timestamp, float] | None = None,
):

    """Main "Transmission & Power" figure with extra log-derived detail.

    In addition to the original current + message-activity view, this
    version can:

      * Overlay vertical markers for parent-search-related log lines
        (handled by add_betterparent_markers).
      * Optionally add a third Y-axis with RSS (dBm) extracted from
        MeshForwarder "rss:" lines.
      * Annotate the title with the last known RSS sample.
    """
    t = df_tel["timestamp"]
    tx_total = pd.to_numeric(df_tel["tx_total"], errors="coerce")
    rx_total = pd.to_numeric(df_tel["rx_total"], errors="coerce")

    i_tel = pd.to_numeric(df_tel["avg_current_uA"], errors="coerce") \
        if "avg_current_uA" in df_tel.columns else None
    i_ppk_row = pd.to_numeric(df_tel["ppk_avg_current_uA"], errors="coerce") \
        if "ppk_avg_current_uA" in df_tel.columns else None

    delta_tx = tx_total.diff().fillna(0)
    delta_rx = rx_total.diff().fillna(0)
    msg_activity = delta_tx + delta_rx

    # Choose a bar width in days (matplotlib date units)
    interval_s = t.diff().dt.total_seconds().median()
    if np.isnan(interval_s) or interval_s <= 0:
        bar_width = (1 / 86400) * 0.8
    else:
        bar_width = (interval_s / 86400) * 0.8

    fig, ax_left = plt.subplots(figsize=(12, 5))

    # Background shading per effective parent
    mark_parent_periods(ax_left, df_tel, alpha=0.35, zorder=0.02)

    # Current traces
    if i_ppk_row is not None and has_data(i_ppk_row):
        ax_left.plot(t, i_ppk_row, label="PPK Avg Current (μA)")

    if i_tel is not None and has_data(i_tel):
        ax_left.plot(
            t,
            i_tel,
            linestyle="--",
            alpha=0.7,
            label="Telemetry Avg Current (μA)",
        )

    if df_ppk is not None and not df_ppk.empty:
        ax_left.plot(
            df_ppk["timestamp"],
            df_ppk["current_uA"],
            alpha=0.4,
            linewidth=0.7,
            label="PPK raw (μA)",
        )

    ax_left.set_ylabel("Current (μA)")
    ax_left.grid(True)

    # Message activity histogram on a second Y-axis
    ax_right = ax_left.twinx()
    ax_right.bar(
        t,
        msg_activity,
        width=bar_width,
        align="center",
        alpha=0.7,
        label="Msg Activity (Δ TX+RX)",
    )
    ax_right.set_ylabel("Message Activity (packets)")

    # Optional RSS trace on a third Y-axis, offset slightly to the right
    ax_rss = None
    if df_rss is not None and not df_rss.empty:
        ax_rss = ax_left.twinx()
        ax_rss.spines["right"].set_position(("axes", 1.12))
        ax_rss.set_ylabel("RSS (dBm)")
        ax_rss.plot(
            df_rss["timestamp"],
            df_rss["rss"],
            marker="o",
            linestyle="None",
            alpha=0.7,
            label="RSS (dBm)",
        )

    # Overlays derived from parent events + log
    add_parent_switch_markers(ax_left, df_events)
    add_betterparent_markers(ax_left, betterparent_times)
    add_noack_markers(ax_left, df_noack)

    # Title (optionally annotated with last RSS)
    title = "Transmission and Power with Parent Periods"
    if last_rss is not None:
        ts_rss, rss_val = last_rss
        try:
            ts_str = ts_rss.strftime("%H:%M:%S")
        except Exception:
            ts_str = str(ts_rss)
        title += f"\nLast RSS: {rss_val:.1f} dBm @ {ts_str}"
    fig.suptitle(title)

    setup_time_axis(ax_left)

    # Combined legend across all axes
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    if ax_rss is not None:
        h3, l3 = ax_rss.get_legend_handles_labels()
        handles += h3
        labels += l3
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


# ========= main ========= #

def main():
    args = parse_args()

    if args.stamp is None and args.ppk is None:
        tel_path, ev_path, token, ppk_path = interactive_select_files()
    else:
        tel_path, ev_path, token = select_esp_pair(args.stamp)
        ppk_path = Path(args.ppk) if args.ppk is not None else None

    df_tel = load_telemetry(tel_path)
    df_events = load_parent_events(ev_path)
    df_tel = build_parent_from_events(df_tel, df_events)

    df_ppk = None
    if ppk_path is not None:
        df_ppk_raw = load_ppk_csv(ppk_path)
        df_ppk = align_ppk_to_telemetry(df_ppk_raw, df_tel)
        df_tel = attach_ppk_averages_to_telemetry(df_tel, df_ppk)

    # ---- log-based annotations: parent search + RSS ----
    
    
    
    
    
    ####
    log_path = Path(f"esp_log_{token}.txt")
    betterparent_times: list[pd.Timestamp] = []
    df_rss: pd.DataFrame | None = None
    last_rss: tuple[pd.Timestamp, float] | None = None
    df_noack: pd.DataFrame | None = None

    if log_path.exists():
        betterparent_times = load_betterparent_times(log_path)
        df_rss, last_rss = load_rss_series(log_path)
        df_noack = load_noack_events(log_path)

        if betterparent_times:
            print(f"[INFO] Found {len(betterparent_times)} PeriodicParentSearch checks in {log_path.name}.")
        else:
            print(f"[INFO] No parent-search check events found in {log_path.name}.")
    
        if df_noack is not None and not df_noack.empty:
            print(f"[INFO] Found {len(df_noack)} MAC NoAck failures in {log_path.name}.")
        else:
            print("[INFO] No MAC NoAck failures found in log.")
    
        if last_rss is not None:
            ts_rss, rss_val = last_rss
            print(f"[INFO] Last RSS in log: {rss_val:.1f} dBm at {ts_rss}")
        else:
            print("[INFO] No 'rss:' lines found in log.")
    else:
        print(f"[INFO] No log file {log_path.name} found; skipping log-derived markers.")

    ####

    plot_transmission_and_power(
        df_tel,
        df_events,
        betterparent_times,
        df_noack=df_noack,
        df_ppk=df_ppk,
        df_rss=df_rss,
        last_rss=last_rss,
    )



if __name__ == "__main__":
    main()
