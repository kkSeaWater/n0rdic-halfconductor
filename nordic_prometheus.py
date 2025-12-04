"""
nordic_oracle.py

Oracle-style "Transmission and Power with Parent Periods" plot for nordic tests.

Files (same timestamp token, e.g. 20251120_144157):

    nordic_telemetry_YYYYMMDD_HHMMSS.csv
    nordic_parent_events_YYYYMMDD_HHMMSS.csv
    nordic_log_YYYYMMDD_HHMMSS.txt   (optional, for BetterParent markers)

Optional PPK2 spreadsheet:

    - New preferred: absolute timestamp + Current(uA)
    - Legacy:        Timestamp(ms) + Current(uA) (relative ms)

Behaviour:

    - If you call with --stamp:
        * Uses nordic_telemetry_<stamp>.csv and nordic_parent_events_<stamp>.csv
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
      - Optional third Y-axis: RSS (dBm) derived from "rss:" in nordic log.
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
        description="nordic oracle-style Transmission & Power plot",
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
    From a filename like 'nordic_telemetry_20251120_144157.csv'
    return '20251120_144157'.
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[-2:])
    return stem


def select_nordic_pair(stamp: str | None):
    """
    Choose telemetry + parent_events files and return:
        (telemetry_path, parent_events_path, token)

    - If stamp is None:
        pick the telemetry file with the **largest** timestamp token.
    - If stamp is provided:
        match any telemetry whose token == stamp OR token.endswith(stamp).
    """
    tele_files = sorted(Path(".").glob("nordic_telemetry_*.csv"))
    if not tele_files:
        raise FileNotFoundError("No nordic_telemetry_*.csv files found in this folder.")

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

    parent_path = Path(f"nordic_parent_events_{token}.csv")
    if not parent_path.exists():
        raise FileNotFoundError(f"Matching parent events file {parent_path} not found.")

    print("Telemetry file   :", tel_path.name)
    print("Parent events    :", parent_path.name)
    return tel_path, parent_path, token


def interactive_select_files():
    """
    Interactive version of select_nordic_pair() that also lets you
    optionally choose a PPK2 CSV from the current directory.

    Returns:
        tel_path, parent_path, token, ppk_path_or_None
    """
    tele_files = sorted(Path(".").glob("nordic_telemetry_*.csv"))
    if not tele_files:
        raise FileNotFoundError("No nordic_telemetry_*.csv files found in this folder.")

    # Choose the newest token
    candidates = [(extract_token_from_name(f), f) for f in tele_files]
    candidates.sort(key=lambda x: x[0])
    token, tel_path = candidates[-1]

    parent_path = Path(f"nordic_parent_events_{token}.csv")
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
    Parse MAC NoAck failures from the nordic log.

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
    Load the nordic_parent_events_*.csv file and add a parsed timestamp column.

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
    """Parse the nordic log for periodic parent-search check events only.

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
    """Extract an RSS (dBm) time-series from the nordic log.

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
        ax.axvline(ts, linestyle=":", linewidth=1.5, color="#005aff", alpha=0.9, label=lbl)
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
            linewidth=1.2,
            color="black",
            alpha=0.8,
            label=label,
        )
        used_label = True

def plot_noack_intensity(df_noack: pd.DataFrame | None, bin_sec: int = 5):
    """
    Plot the intensity of MAC NoAck failures over time.

    We bucket the NoAck events into fixed-size time bins (default 5 seconds)
    and show how many failures occurred in each bin.
    """
    if df_noack is None or df_noack.empty:
        print("[INFO] No MAC NoAck data available for intensity plot.")
        return

    df = df_noack.copy()
    df = df.dropna(subset=["timestamp"])

    # Use the timestamp as index for easy resampling
    df = df.set_index("timestamp")

    # Count how many NoAck events fall into each bin
    rule = f"{bin_sec}S"
    counts = df.resample(rule).size().rename("noack_count")

    if counts.empty or counts.max() == 0:
        print("[INFO] No NoAck events after resampling; skipping intensity plot.")
        return

    counts = counts.reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Bar width in days (matplotlib time units)
    bar_width_days = (bin_sec / 86400.0) * 0.8

    ax.bar(
        counts["timestamp"],
        counts["noack_count"],
        width=bar_width_days,
        align="center",
        alpha=0.8,
    )

    ax.set_title(f"MAC NoAck Failure Intensity (per {bin_sec}s)")
    ax.set_ylabel("NoAck failures per bin")
    ax.set_xlabel("Time")

    setup_time_axis(ax)
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    plt.show()


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
    
def plot_rss_with_filters(
    df_tel: pd.DataFrame,
    df_rss: pd.DataFrame | None,
    betterparent_times=None,
    df_events: pd.DataFrame | None = None,
):
    """
    Plot RSS (dBm) over time with parent backdrop and several rolling
    average filters.

    - Background: shaded by df_tel['effective_parent']
    - Points: raw RSS samples from df_rss
    - Lines: rolling means with different time windows
    """
    if df_rss is None or df_rss.empty:
        print("[INFO] No RSS data available; skipping RSS-filter plot.")
        return

    # Clean and index by timestamp
    rss = df_rss.copy()
    rss = rss.dropna(subset=["timestamp", "rss"]).sort_values("timestamp")
    rss = rss.set_index("timestamp")

    if rss.empty:
        print("[INFO] RSS DataFrame empty after cleaning; skipping plot.")
        return

    # Time-based rolling windows
    # You can tweak these windows if you like
    windows = [
        ("5s",  "RSS rolling 5 s"),
        ("15s", "RSS rolling 15 s"),
        ("60s", "RSS rolling 60 s"),
    ]

    for win, _ in windows:
        rss[f"roll_{win}"] = rss["rss"].rolling(win, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))

    # Parent backdrop (uses df_tel['effective_parent'])
    mark_parent_periods(ax, df_tel, alpha=0.35, zorder=0.02)

    # Raw RSS points
    ax.plot(
        rss.index,
        rss["rss"],
        linestyle="None",
        marker="o",
        alpha=0.4,
        label="RSS raw",
    )

    # Rolling averages
    for win, label in windows:
        col = f"roll_{win}"
        ax.plot(
            rss.index,
            rss[col],
            label=label,
        )

    ax.set_ylabel("RSS (dBm)")
    ax.set_title("RSS with Rolling Averages and Parent Periods")
    ax.grid(True, axis="y", alpha=0.4)

    # Optional markers for parent switches & parent-search ticks
    if df_events is not None and not df_events.empty:
        add_parent_switch_markers(ax, df_events)
    if betterparent_times is not None:
        add_betterparent_markers(ax, betterparent_times)

    setup_time_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_noack_vs_activity(
    df_tel: pd.DataFrame,
    df_noack: pd.DataFrame | None,
    df_events: pd.DataFrame,
    betterparent_times=None,
    bin_sec: int = 5,
):
    """
    Compare MAC NoAck errors to message activity (ΔTX + ΔRX).

    - Parent backdrop (effective_parent)
    - Bars: total message activity per bin
    - Line: NoAck count per bin (same bins)
    """
    if df_tel.empty:
        print("[INFO] Telemetry DataFrame empty; skipping NoAck vs activity plot.")
        return
    if df_noack is None or df_noack.empty:
        print("[INFO] No MAC NoAck data; skipping NoAck vs activity plot.")
        return

    # --- Build message-activity series from telemetry ---
    t = df_tel["timestamp"]

    def delta(col):
        if col not in df_tel.columns:
            return None
        s = pd.to_numeric(df_tel[col], errors="coerce")
        return s.diff().fillna(0)

    d_tx = delta("tx_total")
    d_rx = delta("rx_total")
    if d_tx is None or d_rx is None:
        print("[INFO] tx_total or rx_total missing; cannot compute message activity.")
        return

    msg_activity = (d_tx + d_rx).fillna(0)

    df_act = pd.DataFrame({"timestamp": t, "msg_activity": msg_activity})
    df_act = df_act.dropna(subset=["timestamp"]).set_index("timestamp")

    rule = f"{bin_sec}s"
    act_binned = df_act["msg_activity"].resample(rule).sum()

    # --- Build NoAck-count series on same bins ---
    df_n = df_noack.dropna(subset=["timestamp"]).copy().set_index("timestamp")
    noack_binned = df_n.resample(rule).size().rename("noack_count")

    # Align indexes
    combined_index = act_binned.index.union(noack_binned.index)
    act_binned = act_binned.reindex(combined_index, fill_value=0)
    noack_binned = noack_binned.reindex(combined_index, fill_value=0)

    # --- Plot ---
    fig, ax_left = plt.subplots(figsize=(12, 4))

    # Parent backdrop from full telemetry
    mark_parent_periods(ax_left, df_tel, alpha=0.35, zorder=0.02)

    # Bars = message activity
    bar_width_days = (bin_sec / 86400.0) * 0.8
    ax_left.bar(
        combined_index,
        act_binned,
        width=bar_width_days,
        align="center",
        alpha=0.7,
        label=f"Msg Activity (ΔTX+ΔRX per {bin_sec}s)",
    )
    ax_left.set_ylabel("Messages per bin")
    ax_left.grid(True, axis="y", alpha=0.4)

    # Right axis = NoAck count per bin
    ax_right = ax_left.twinx()
    ax_right.plot(
        combined_index,
        noack_binned,
        drawstyle="steps-mid",
        marker="o",
        linestyle="-",
        alpha=0.9,
        label=f"NoAck count per {bin_sec}s",
    )
    ax_right.set_ylabel("NoAck failures per bin")

    # Reuse vertical markers
    add_parent_switch_markers(ax_left, df_events)
    if betterparent_times is not None:
        add_betterparent_markers(ax_left, betterparent_times)
    add_noack_markers(ax_left, df_noack)

    fig.suptitle(f"NoAck Errors vs Message Activity (bin = {bin_sec}s)")
    setup_time_axis(ax_left)

    # Combined legend
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_ppk_current(
    df_ppk: pd.DataFrame | None,
    df_ppk_raw: pd.DataFrame | None = None,
):
    """
    PPK current vs time, with 3 different rolling-average filters.

    - Prefer the aligned PPK DataFrame (df_ppk) if it has data.
    - Fall back to the raw PPK DataFrame (df_ppk_raw) if alignment
      dropped everything.
    - Plot:
        * Raw current
        * Rolling 5 s
        * Rolling 15 s
        * Rolling 60 s
    """
    # Decide which dataframe to use
    df = None
    label_base = "PPK current"

    if df_ppk is not None and not df_ppk.empty:
        df = df_ppk.copy()
        label_base = "PPK current (aligned)"
    elif df_ppk_raw is not None and not df_ppk_raw.empty:
        df = df_ppk_raw.copy()
        label_base = "PPK current (raw)"

    if df is None or df.empty:
        print("[INFO] No PPK data; skipping PPK current-only plot.")
        return

    # Use timestamp as the index for time-based rolling windows
    df = df.sort_values("timestamp").set_index("timestamp")

    # Define rolling windows (same style as RSS plot)
    windows = [
        ("5s",  "Current rolling 5 s"),
        ("15s", "Current rolling 15 s"),
        ("60s", "Current rolling 60 s"),
    ]

    # Compute rolling means
    for win, _ in windows:
        df[f"roll_{win}"] = df["current_uA"].rolling(win, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))

    # Raw current (thin, more transparent)
    ax.plot(
        df.index,
        df["current_uA"],
        linewidth=0.6,
        alpha=0.4,
        label=f"{label_base} raw",
    )

    # Rolling-average lines
    for win, lbl in windows:
        col = f"roll_{win}"
        ax.plot(
            df.index,
            df[col],
            linewidth=1.2,
            label=lbl,
        )

    ax.set_title("PPK Current vs Time (with Rolling Averages)")
    ax.set_ylabel("Current (μA)")
    ax.set_xlabel("Time")
    ax.grid(True, axis="y", alpha=0.4)

    # Same time-formatting as the other plots
    setup_time_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    
def plot_current_vs_activity(
    df_tel: pd.DataFrame,
    df_ppk: pd.DataFrame | None,
    df_ppk_raw: pd.DataFrame | None = None,
    bin_sec: int = 5,
):
    """
    Plot PPK current (with rolling averages) together with message activity.

    Left Y-axis:
        - Raw current
        - Rolling 5 s, 15 s, 60 s

    Right Y-axis:
        - Message activity (ΔTX + ΔRX) binned into `bin_sec`-second buckets.

    The x-axis is **restricted to the time span where PPK has data**
    so the plot doesn't look completely compressed.
    """
    if df_tel is None or df_tel.empty:
        print("[INFO] Telemetry DataFrame empty; skipping current vs activity plot.")
        return

    # -------- choose which PPK dataframe to use -------- #
    df_cur = None
    label_base = "PPK current"

    if df_ppk is not None and not df_ppk.empty:
        df_cur = df_ppk.copy()
        label_base = "PPK current (aligned)"
    elif df_ppk_raw is not None and not df_ppk_raw.empty:
        df_cur = df_ppk_raw.copy()
        label_base = "PPK current (raw)"

    if df_cur is None or df_cur.empty:
        print("[INFO] No PPK data; skipping current vs activity plot.")
        return

    # Ensure timestamp is datetime and sorted
    df_cur = df_cur.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_cur = df_cur.set_index("timestamp")

    # This is the window we care about (so we don't get a 12-hour x-axis)
    t_start = df_cur.index.min()
    t_end   = df_cur.index.max()

    # -------- rolling averages on current -------- #
    windows = [
        ("5s",  "Current rolling 5 s"),
        ("15s", "Current rolling 15 s"),
        ("60s", "Current rolling 60 s"),
    ]

    for win, _ in windows:
        df_cur[f"roll_{win}"] = df_cur["current_uA"].rolling(win, min_periods=1).mean()

    # -------- message activity from telemetry -------- #
    t_tel = df_tel["timestamp"]
    tx_total = pd.to_numeric(df_tel["tx_total"], errors="coerce")
    rx_total = pd.to_numeric(df_tel["rx_total"], errors="coerce")

    d_tx = tx_total.diff().clip(lower=0)
    d_rx = rx_total.diff().clip(lower=0)
    msg_activity = (d_tx + d_rx).fillna(0)

    df_act = pd.DataFrame(
        {
            "timestamp": t_tel,
            "msg_activity": msg_activity,
        }
    )
    df_act = df_act.dropna(subset=["timestamp"]).set_index("timestamp")

    rule = f"{bin_sec}s"
    msg_binned = df_act["msg_activity"].resample(rule).sum()

    # *** CRUCIAL FIX: clip activity to the PPK time window ***
    msg_binned = msg_binned.loc[t_start:t_end]

    # -------- plotting -------- #
    fig, ax_left = plt.subplots(figsize=(12, 4))

    # Optional parent backdrop
    try:
        mark_parent_periods(ax_left, df_tel, alpha=0.25, zorder=0.02)
    except Exception:
        pass

    # Current: raw
    ax_left.plot(
        df_cur.index,
        df_cur["current_uA"],
        linewidth=0.6,
        alpha=0.4,
        label=f"{label_base} raw",
    )

    # Current: rolling averages
    for win, lbl in windows:
        col = f"roll_{win}"
        ax_left.plot(
            df_cur.index,
            df_cur[col],
            linewidth=1.2,
            label=lbl,
        )

    ax_left.set_ylabel("Current (μA)")
    ax_left.set_xlabel("Time")
    ax_left.grid(True, axis="y", alpha=0.4)

    # Message activity on second axis
    ax_right = ax_left.twinx()
    bar_width_days = (bin_sec / 86400.0) * 0.8

    ax_right.bar(
        msg_binned.index,
        msg_binned.values,
        width=bar_width_days,
        align="center",
        alpha=0.5,
        label=f"Msg activity per {bin_sec}s (ΔTX+ΔRX)",
    )
    ax_right.set_ylabel("Message Activity (packets)")

    ax_left.set_title(
        f"PPK Current (raw + rolling) and Message Activity (bin = {bin_sec}s)"
    )

    setup_time_axis(ax_left)

    # *** Also force x-limits to the PPK window ***
    ax_left.set_xlim(t_start, t_end)

    # Combined legend
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_current_rss_noack(
    df_tel: pd.DataFrame,
    df_ppk: pd.DataFrame | None,
    df_ppk_raw: pd.DataFrame | None,
    df_rss: pd.DataFrame | None,
    df_noack: pd.DataFrame | None,
    bin_sec: int = 5,
):
    """
    Combined plot:

        - Current (rolling 5 s) from PPK
        - RSS (dBm) from df_rss
        - MAC NoAck intensity (counts per `bin_sec` seconds)

    Assumes PPK timestamps have already been shifted so that
    PPK start == telemetry start (your current assumption).
    """
    # ---------- choose which PPK dataframe to use ---------- #
    df_cur = None
    label_base = "PPK current"

    if df_ppk is not None and not df_ppk.empty:
        df_cur = df_ppk.copy()
        label_base = "PPK current (aligned)"
    elif df_ppk_raw is not None and not df_ppk_raw.empty:
        df_cur = df_ppk_raw.copy()
        label_base = "PPK current (raw)"

    if df_cur is None or df_cur.empty:
        print("[INFO] No PPK data; skipping current/RSS/NoAck plot.")
        return

    # Clean + index by timestamp
    df_cur = df_cur.dropna(subset=["timestamp"]).sort_values("timestamp")
    df_cur = df_cur.set_index("timestamp")

    # Time window driven by PPK (since that’s the “expensive” trace)
    t_start = df_cur.index.min()
    t_end   = df_cur.index.max()

    # ---------- current: rolling 5 seconds ---------- #
    df_cur["roll_5s"] = df_cur["current_uA"].rolling("5s", min_periods=1).mean()

    # ---------- RSS handling ---------- #
    rss_clip = None
    if df_rss is not None and not df_rss.empty:
        rss = df_rss.copy()
        rss = rss.dropna(subset=["timestamp"]).sort_values("timestamp")
        rss = rss.set_index("timestamp")
        rss_clip = rss.loc[t_start:t_end]

    # ---------- NoAck intensity handling ---------- #
    counts = None
    if df_noack is not None and not df_noack.empty:
        df_na = df_noack.copy()
        df_na = df_na.dropna(subset=["timestamp"]).set_index("timestamp")

        rule = f"{bin_sec}S"
        counts_series = df_na.resample(rule).size().rename("noack_count")
        counts_series = counts_series.loc[t_start:t_end]

        if not counts_series.empty and counts_series.max() > 0:
            counts = counts_series.reset_index()

    # ---------- plotting ---------- #
    fig, ax_cur = plt.subplots(figsize=(12, 4))

    # Optional parent backdrop
    try:
        mark_parent_periods(ax_cur, df_tel, alpha=0.25, zorder=0.02)
    except Exception:
        pass

    # Current (rolling 5 s)
    ax_cur.plot(
        df_cur.index,
        df_cur["roll_5s"],
        linewidth=1.2,
        label=f"{label_base} rolling 5 s",
    )
    ax_cur.set_ylabel("Current (μA)")
    ax_cur.set_xlabel("Time")
    ax_cur.grid(True, axis="y", alpha=0.4)

    # ---------- RSS on 2nd axis ---------- #
    ax_rss = ax_cur.twinx()
    if rss_clip is not None and not rss_clip.empty:
        ax_rss.plot(
            rss_clip.index,
            rss_clip["rss"],
            linestyle="--",
            markersize=3.0,
            marker="o",
            label="RSS (dBm)",
            color="red"
        )
    ax_rss.set_ylabel("RSS (dBm)")

    # ---------- NoAck intensity on 3rd axis (right, shifted) ---------- #
    ax_na = ax_cur.twinx()
    ax_na.spines["right"].set_position(("axes", 1.12))  # shift a bit to the right
    if counts is not None:
        bar_width_days = (bin_sec / 86400.0) * 0.8
        ax_na.bar(
            counts["timestamp"],
            counts["noack_count"],
            width=bar_width_days,
            align="center",
            alpha=0.4,
            label=f"NoAck per {bin_sec}s",
        )
    ax_na.set_ylabel("NoAck failures / bin")

    ax_cur.set_title(
        f"Current (rolling 5 s), RSS and NoAck Intensity (bin = {bin_sec}s)"
    )

    setup_time_axis(ax_cur)
    ax_cur.set_xlim(t_start, t_end)

    # ---------- combined legend ---------- #
    h1, l1 = ax_cur.get_legend_handles_labels()
    h2, l2 = ax_rss.get_legend_handles_labels()
    h3, l3 = ax_na.get_legend_handles_labels()
    handles = h1 + h2 + h3
    labels = l1 + l2 + l3
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    
def plot_telemetry_detail(
    df_tel: pd.DataFrame,
    df_events: pd.DataFrame,
    betterparent_times=None,
    df_noack: pd.DataFrame | None = None,
    df_rss: pd.DataFrame | None = None,
):
    """
    Telemetry-focused figure with parent backdrop.

    Shows per-interval deltas of key counters (TX, RX, retries, errors)
    over time, shaded by effective_parent, and optionally overlays RSS.
    """
    if df_tel.empty:
        print("[INFO] Telemetry DataFrame is empty; skipping telemetry detail plot.")
        return

    t = df_tel["timestamp"]

    # Helper to get per-interval deltas if the column exists
    def delta(colname):
        if colname not in df_tel.columns:
            return None
        s = pd.to_numeric(df_tel[colname], errors="coerce")
        return s.diff().fillna(0)

    d_tx_total   = delta("tx_total")
    d_rx_total   = delta("rx_total")
    d_tx_retry   = delta("tx_retry")
    d_tx_err_cca = delta("tx_err_cca")
    d_rx_err_fcs = delta("rx_err_fcs")

    fig, ax_main = plt.subplots(figsize=(12, 5))

    # Background shading per effective parent (same style as main plot)
    mark_parent_periods(ax_main, df_tel, alpha=0.35, zorder=0.02)

    # Plot the available delta series on the main (left) axis
    if d_tx_total is not None and has_data(d_tx_total):
        ax_main.plot(t, d_tx_total, label="ΔTX total")

    if d_rx_total is not None and has_data(d_rx_total):
        ax_main.plot(t, d_rx_total, label="ΔRX total", linestyle="--", alpha=0.8)

    if d_tx_retry is not None and has_data(d_tx_retry):
        ax_main.plot(t, d_tx_retry, label="ΔTX retry", linestyle="-.", alpha=0.9)

    if d_tx_err_cca is not None and has_data(d_tx_err_cca):
        ax_main.plot(t, d_tx_err_cca, label="ΔTX err CCA", linestyle=":", alpha=0.9)

    if d_rx_err_fcs is not None and has_data(d_rx_err_fcs):
        ax_main.plot(t, d_rx_err_fcs, label="ΔRX err FCS", linewidth=1.2)

    ax_main.set_ylabel("Per-interval counts")
    ax_main.grid(True, axis="y", alpha=0.4)

    # Optional RSS on a secondary Y-axis (right)
    ax_rss = None
    if df_rss is not None and not df_rss.empty:
        ax_rss = ax_main.twinx()
        ax_rss.set_ylabel("RSS (dBm)")
        ax_rss.plot(
            df_rss["timestamp"],
            df_rss["rss"],
            marker="o",
            linestyle="None",
            alpha=0.7,
            label="RSS (dBm)",
        )

    # Reuse the same vertical markers as the main plot
    add_parent_switch_markers(ax_main, df_events)
    if betterparent_times is not None:
        add_betterparent_markers(ax_main, betterparent_times)
    add_noack_markers(ax_main, df_noack)

    fig.suptitle("Telemetry Counters with Parent Periods")

    setup_time_axis(ax_main)

    # Combined legend across axes
    h1, l1 = ax_main.get_legend_handles_labels()
    handles, labels = list(h1), list(l1)
    if ax_rss is not None:
        h2, l2 = ax_rss.get_legend_handles_labels()
        handles += list(h2)
        labels += list(l2)
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_txerror_vs_rss(
    df_tel: pd.DataFrame,
    df_rss: pd.DataFrame | None,
    df_events: pd.DataFrame,
    betterparent_times=None,
):
    """
    Plot TX error (from tx_err_cca) over time together with RSS (dBm).

    Left Y-axis:
        - Per-interval TX error counts (Δ of tx_err_cca)

    Right Y-axis:
        - RSS values from the log (with a 5 s rolling mean)

    Background:
        - Shaded by effective parent, same style as other plots.
    """
    if df_tel.empty:
        print("[INFO] Telemetry DataFrame empty; skipping TX error vs RSS plot.")
        return
    if df_rss is None or df_rss.empty:
        print("[INFO] No RSS data; skipping TX error vs RSS plot.")
        return

    if "tx_err_cca" not in df_tel.columns:
        print("[INFO] No 'tx_err_cca' column in telemetry; skipping TX error vs RSS plot.")
        return

    # ----- build TX error delta series ----- #
    t = df_tel["timestamp"]
    tx_err_raw = pd.to_numeric(df_tel["tx_err_cca"], errors="coerce")
    d_tx_err = tx_err_raw.diff()

    # Replace NaNs from the first diff with 0
    d_tx_err = d_tx_err.fillna(0)

    # If literally everything is NaN, *then* bail
    if d_tx_err.isna().all():
        print("[INFO] TX error deltas are all NaN; skipping TX error vs RSS plot.")
        return

    # NOTE: we no longer skip if it's all zeros; that will just show a flat line at 0

    # ----- prepare RSS series ----- #
    rss = df_rss.copy()
    rss = rss.dropna(subset=["timestamp", "rss"]).sort_values("timestamp")
    if rss.empty:
        print("[INFO] RSS DataFrame empty after cleaning; skipping TX error vs RSS plot.")
        return

    rss = rss.set_index("timestamp")
    rss["rss_roll_5s"] = rss["rss"].rolling("5s", min_periods=1).mean()

    # ----- plotting ----- #
    fig, ax_err = plt.subplots(figsize=(12, 4))

    # Parent backdrop
    try:
        mark_parent_periods(ax_err, df_tel, alpha=0.35, zorder=0.02)
    except Exception:
        pass

    # Left axis: TX error per interval
    ax_err.plot(
        t,
        d_tx_err,
        linestyle="-",
        marker="o",
        alpha=0.9,
        label="ΔTX err CCA",
    )
    ax_err.set_ylabel("TX errors per interval")
    ax_err.grid(True, axis="y", alpha=0.4)

    # Right axis: RSS
    ax_rss = ax_err.twinx()
    ax_rss.set_ylabel("RSS (dBm)")

    # Raw RSS points
    ax_rss.plot(
        rss.index,
        rss["rss"],
        linestyle="None",
        marker="o",
        alpha=0.35,
        label="RSS raw",
    )

    # Rolling 5 s smoothed RSS
    ax_rss.plot(
        rss.index,
        rss["rss_roll_5s"],
        linestyle="-",
        alpha=0.9,
        label="RSS rolling 5 s",
    )

    # Reuse your existing markers
    add_parent_switch_markers(ax_err, df_events)
    if betterparent_times is not None:
        add_betterparent_markers(ax_err, betterparent_times)

    ax_err.set_title("TX Error (Δtx_err_cca) vs RSS with Parent Periods")
    setup_time_axis(ax_err)

    # Combined legend
    h1, l1 = ax_err.get_legend_handles_labels()
    h2, l2 = ax_rss.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    
def plot_noack_vs_rss(
    df_tel: pd.DataFrame,
    df_noack: pd.DataFrame | None,
    df_rss: pd.DataFrame | None,
    df_events: pd.DataFrame,
    betterparent_times=None,
    bin_sec: int = 5,
):
    """
    Compare MAC NoAck (ACK failures) to RSS over time.

    Left Y-axis:
        - NoAck count per time bin (e.g. 5 seconds)

    Right Y-axis:
        - RSS (raw) and a 5-second rolling mean

    Background:
        - Shaded by effective_parent (same style as other plots)
        - Parent-switch markers, PeriodicParentSearch markers
    """
    if df_noack is None or df_noack.empty:
        print("[INFO] No MAC NoAck data; skipping NoAck vs RSS plot.")
        return
    if df_rss is None or df_rss.empty:
        print("[INFO] No RSS data; skipping NoAck vs RSS plot.")
        return
    if df_tel.empty:
        print("[INFO] Telemetry DataFrame empty; skipping NoAck vs RSS plot.")
        return

    # ---- 1. Prepare NoAck counts per bin ---- #
    df_na = df_noack.copy()
    df_na = df_na.dropna(subset=["timestamp"]).set_index("timestamp")

    # IMPORTANT: use lowercase 's', not 'S', to avoid FutureWarning
    rule = f"{bin_sec}s"
    counts_series = df_na.resample(rule).size().rename("noack_count")

    if counts_series.empty or counts_series.sum() == 0:
        print("[INFO] No NoAck events after resampling; NoAck vs RSS will show only zeros.")
        # still proceed; you'll just see a flat 0 line

    df_counts = counts_series.to_frame()
    df_counts["timestamp"] = df_counts.index

    # ---- 2. Prepare RSS series (with rolling mean) ---- #
    rss = df_rss.copy()
    rss = rss.dropna(subset=["timestamp", "rss"]).sort_values("timestamp")
    if rss.empty:
        print("[INFO] RSS DataFrame empty after cleaning; skipping NoAck vs RSS plot.")
        return

    rss = rss.set_index("timestamp")
    rss["rss_roll_5s"] = rss["rss"].rolling("5s", min_periods=1).mean()

    # ---- 3. Plotting ---- #
    fig, ax_left = plt.subplots(figsize=(12, 4))

    # Parent backdrop from full telemetry
    try:
        mark_parent_periods(ax_left, df_tel, alpha=0.35, zorder=0.02)
    except Exception:
        pass

    # Bar width in days (matplotlib time units)
    bar_width_days = (bin_sec / 86400.0) * 0.8

    # Left axis: NoAck counts per bin
    ax_left.bar(
        df_counts["timestamp"],
        df_counts["noack_count"],
        width=bar_width_days,
        align="center",
        alpha=0.7,
        label=f"NoAck count per {bin_sec}s",
    )
    ax_left.set_ylabel(f"NoAck failures / {bin_sec}s")
    ax_left.grid(True, axis="y", alpha=0.4)

    # Right axis: RSS
    ax_right = ax_left.twinx()
    ax_right.set_ylabel("RSS (dBm)")

    # Raw RSS points
    ax_right.plot(
        rss.index,
        rss["rss"],
        linestyle="None",
        marker="o",
        alpha=0.3,
        label="RSS raw",
    )

    # Rolling 5s smoothed RSS
    ax_right.plot(
        rss.index,
        rss["rss_roll_5s"],
        linestyle="-",
        alpha=0.9,
        label="RSS rolling 5 s",
    )

    # Parent-switch markers + PeriodicParentSearch markers
    add_parent_switch_markers(ax_left, df_events)
    if betterparent_times is not None:
        add_betterparent_markers(ax_left, betterparent_times)

    ax_left.set_title(f"MAC NoAck (ACK failures) vs RSS with Parent Periods (bin = {bin_sec}s)")
    setup_time_axis(ax_left)

    # Combined legend
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()
    
    def plot_txretry_vs_rss(
    df_tel: pd.DataFrame,
    df_rss: pd.DataFrame | None,
    df_events: pd.DataFrame,
    betterparent_times=None,):
        if df_tel.empty:
            print("[INFO] Telemetry DataFrame empty; skipping TX retry vs RSS plot.")
            return
        if df_rss is None or df_rss.empty:
            print("[INFO] No RSS data; skipping TX retry vs RSS plot.")
            return
        if "tx_retry" not in df_tel.columns:
            print("[INFO] No 'tx_retry' column in telemetry; skipping TX retry vs RSS plot.")
            return

        # ---- 1. Build ΔTX retry series ---- #
    t = df_tel["timestamp"]
    tx_retry_raw = pd.to_numeric(df_tel["tx_retry"], errors="coerce")
    d_tx_retry = tx_retry_raw.diff().fillna(0)

    # guard against counter wrap/resets (negative deltas)
    d_tx_retry = d_tx_retry.mask(d_tx_retry < 0, other=0)

    # ---- 2. Prepare RSS series (with rolling mean) ---- #
    rss = df_rss.copy()
    rss = rss.dropna(subset=["timestamp", "rss"]).sort_values("timestamp")
    if rss.empty:
        print("[INFO] RSS DataFrame empty after cleaning; skipping TX retry vs RSS plot.")
        return

    rss = rss.set_index("timestamp")
    # 5-second rolling average to smooth the RSS
    rss["rss_roll_5s"] = rss["rss"].rolling("5s", min_periods=1).mean()

    # ---- 3. Plotting ---- #
    fig, ax_retry = plt.subplots(figsize=(12, 4))

    # Parent backdrop
    try:
        mark_parent_periods(ax_retry, df_tel, alpha=0.35, zorder=0.02)
    except Exception:
        pass

    # Left axis: ΔTX retry
    ax_retry.plot(
        t,
        d_tx_retry,
        linestyle="-",
        marker="o",
        alpha=0.9,
        label="ΔTX retry",
    )
    ax_retry.set_ylabel("TX retries per interval")
    ax_retry.grid(True, axis="y", alpha=0.4)

    # Right axis: RSS
    ax_rss = ax_retry.twinx()
    ax_rss.set_ylabel("RSS (dBm)")

    # Raw RSS points
    ax_rss.plot(
        rss.index,
        rss["rss"],
        linestyle="None",
        marker="o",
        alpha=0.3,
        label="RSS raw",
    )

    # Rolling 5 s smoothed RSS
    ax_rss.plot(
        rss.index,
        rss["rss_roll_5s"],
        linestyle="-",
        alpha=0.9,
        label="RSS rolling 5 s",
    )

    # Same vertical markers as other plots
    add_parent_switch_markers(ax_retry, df_events)
    if betterparent_times is not None:
        add_betterparent_markers(ax_retry, betterparent_times)

    ax_retry.set_title("ΔTX retry vs RSS with Parent Periods")
    setup_time_axis(ax_retry)

    # Combined legend (both axes)
    h1, l1 = ax_retry.get_legend_handles_labels()
    h2, l2 = ax_rss.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
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
        tel_path, ev_path, token = select_nordic_pair(args.stamp)
        ppk_path = Path(args.ppk) if args.ppk is not None else None

    df_tel = load_telemetry(tel_path)
    df_events = load_parent_events(ev_path)
    df_tel = build_parent_from_events(df_tel, df_events)

    # --- NEW: keep BOTH raw and aligned PPK ---
    df_ppk_raw: pd.DataFrame | None = None
    df_ppk: pd.DataFrame | None = None

        # --- PPK: re-anchor so PPK start == telemetry start --- #
    df_ppk_raw = None
    df_ppk = None
    if ppk_path is not None:
        df_ppk_raw = load_ppk_csv(ppk_path)

        if df_ppk_raw is not None and not df_ppk_raw.empty:
            # Sort to be safe
            df_ppk_raw = df_ppk_raw.sort_values("timestamp").reset_index(drop=True)

            # Assume PPK recording starts at the SAME moment as telemetry
            t0_tel = df_tel["timestamp"].iloc[0]
            t0_ppk = df_ppk_raw["timestamp"].iloc[0]

            # Shift all PPK samples so that t0_ppk → t0_tel
            df_ppk_raw["timestamp"] = t0_tel + (df_ppk_raw["timestamp"] - t0_ppk)

            # Now trim to telemetry window
            df_ppk = align_ppk_to_telemetry(df_ppk_raw, df_tel)
            df_tel = attach_ppk_averages_to_telemetry(df_tel, df_ppk)



    
    
    
    
    
    ####
    log_path = Path(f"nordic_log_{token}.txt")
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
    
        # Extra figure: telemetry counters vs time, with parent backdrop
    plot_telemetry_detail(
        df_tel,
        df_events,
        betterparent_times=betterparent_times,
        df_noack=df_noack,
        df_rss=df_rss,
    )

    plot_rss_with_filters(
        df_tel,
        df_rss,
        betterparent_times=betterparent_times,
        df_events=df_events,
    )
    
    plot_noack_vs_activity(
        df_tel,
        df_noack,
        df_events,
        betterparent_times=betterparent_times,
        bin_sec=5,
    )

        # Extra figure: how intense are the MAC NoAck failures over time?
    plot_noack_intensity(df_noack, bin_sec=5)
    
    plot_ppk_current(df_ppk, df_ppk_raw)
    
    plot_current_vs_activity(df_tel, df_ppk, df_ppk_raw, bin_sec=5)
    
    plot_current_rss_noack(df_tel, df_ppk, df_ppk_raw, df_rss, df_noack,bin_sec=5)

    plot_txerror_vs_rss(
        df_tel,
        df_rss,
        df_events,
        betterparent_times=betterparent_times,
    )
    
    plot_noack_vs_rss(
        df_tel,
        df_noack,
        df_rss,
        df_events,
        betterparent_times=betterparent_times,
        bin_sec=5,
    )
    
    plot_txretry_vs_rss(
    df_tel,
    df_rss,
    df_events,
    betterparent_times=betterparent_times,)



if __name__ == "__main__":
    main()
