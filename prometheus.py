# build_cutoff_dataset.py
# This version asks you for a folder path, scans it for telemetry CSV files,
# lets you pick one, then builds:
#   1) <name>_cutoff_events.csv
#   2) <name>_risk_features_K15s.csv

import os, glob, pandas as pd, numpy as np
from pathlib import Path

# ---------------- small helpers ----------------

def parse_ts(x):
    if pd.isna(x): return pd.NaT
    try:
        return pd.to_datetime(float(str(x).strip()), unit="s")
    except Exception:
        return pd.to_datetime(str(x).strip(), errors="coerce")

def first_diff(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.diff().clip(lower=0)

def ema(s, span):
    s = pd.to_numeric(s, errors="coerce")
    return s.ewm(span=span, adjust=False, min_periods=1).mean()

def rstd(s, window):
    s = pd.to_numeric(s, errors="coerce")
    return s.rolling(window=window, min_periods=2).std().fillna(0)

def pick_col(df, names, default=None):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([default]*len(df), index=df.index)

# ---------------- main logic ----------------

def load_telemetry(csv_path):
    df = pd.read_csv(csv_path)
    for c in ["timestamp","time","ts","datetime","DateTime","Datetime"]:
        if c in df.columns:
            df["timestamp"] = df[c].apply(parse_ts)
            break
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["state_norm"] = pick_col(df,["state","State","role","Role"],"").astype(str).str.lower()
    df["parent_rloc16"] = pick_col(df,["parent_rloc16","parent","parent_rloc","ParentRLOC16"])

    df["RTT_ms"]  = pd.to_numeric(pick_col(df,["RTT_ms","rtt_ms","RTT","rtt"]), errors="coerce")
    df["LQI_in"]  = pd.to_numeric(pick_col(df,["LQI_in","lqi_in","LQI","lqi"]), errors="coerce")
    df["tx_total"]= pd.to_numeric(pick_col(df,["tx_total","TxTotal","TX_Total"]), errors="coerce")
    df["tx_retry"]= pd.to_numeric(pick_col(df,["tx_retry","TxRetry","TX_Retry"]), errors="coerce")
    df["cca_err"] = pd.to_numeric(pick_col(df,["cca_err","CCA_Err","TxCCAError"]), errors="coerce")
    df["rx_total"]= pd.to_numeric(pick_col(df,["rx_total","RxTotal","RX_Total"]), errors="coerce")
    df["rx_fcs"]  = pd.to_numeric(pick_col(df,["rx_fcs_err","RxFcsError","rx_fcs"]), errors="coerce")
    df["age_s"]   = pd.to_numeric(pick_col(df,["age_s","Age_s","age"]), errors="coerce")
    return df

def detect_from_state(df):
    events = []
    st_now = df["state_norm"].fillna("")
    st_prev = st_now.shift(1).fillna("")
    ts = df["timestamp"]

    for t in ts[(st_now=="detached")&(st_prev!="detached")]:
        events.append({"ts":t,"type":"detached_start"})
    for t in ts[(st_now!="detached")&(st_prev=="detached")]:
        events.append({"ts":t,"type":"reattached"})

    parent = df["parent_rloc16"].astype(str).fillna("")
    parent_prev = parent.shift(1).fillna(parent.iloc[0])
    for t in ts[(parent!=parent_prev)&(st_now!="detached")]:
        events.append({"ts":t,"type":"parent_switch"})
    return sorted(events, key=lambda e:e["ts"])

def build_cutoff_table(df):
    evs = detect_from_state(df)
    rows=[]; last_det=None; last_idx=None
    for e in evs:
        if e["type"]=="detached_start":
            last_det=e["ts"]; last_idx=len(rows)
            rows.append({"ts_event":e["ts"],"event_type":"detached_start",
                         "reattach_ts":pd.NaT,"reattach_s":np.nan})
        elif e["type"]=="reattached" and last_det is not None and last_idx is not None:
            rt=(e["ts"]-last_det).total_seconds()
            if pd.isna(rows[last_idx]["reattach_ts"]):
                rows[last_idx]["reattach_ts"]=e["ts"]; rows[last_idx]["reattach_s"]=round(rt,3)
        elif e["type"]=="parent_switch":
            rows.append({"ts_event":e["ts"],"event_type":"parent_switch",
                         "reattach_ts":pd.NaT,"reattach_s":np.nan})
    return pd.DataFrame(rows)

def build_features(df,cutoff_df,K=15):
    dtx=first_diff(df["tx_total"]); dtxr=first_diff(df["tx_retry"]); dcca=first_diff(df["cca_err"])
    drx=first_diff(df["rx_total"]); drxf=first_diff(df["rx_fcs"])
    with np.errstate(divide="ignore",invalid="ignore"):
        df["tx_retry_rate"]=(dtxr/dtx.replace(0,np.nan)).fillna(0)
        df["cca_err_rate"]=(dcca/dtx.replace(0,np.nan)).fillna(0)
        df["rx_fcs_rate"]=(drxf/drx.replace(0,np.nan)).fillna(0)
    df["RTT_ms_ema"]=ema(df["RTT_ms"],5)
    df["RTT_ms_vol"]=rstd(df["RTT_ms"],10)
    df["LQI_in_ema"]=ema(df["LQI_in"],5)
    df["Age_s"]=pd.to_numeric(df["age_s"],errors="coerce").fillna(method="ffill").fillna(0)
    events=[pd.Timestamp(t) for t in cutoff_df["ts_event"]] if not cutoff_df.empty else []
    def label(t):
        for et in events:
            dt=(et-t).total_seconds()
            if 0<=dt<=K: return 1
        return 0
    df[f"label_will_cutoff_{K}s"]=df["timestamp"].apply(label)
    cols=["timestamp","state_norm","parent_rloc16","RTT_ms","LQI_in","tx_total","tx_retry",
          "cca_err","rx_total","rx_fcs","tx_retry_rate","cca_err_rate","rx_fcs_rate",
          "RTT_ms_ema","RTT_ms_vol","LQI_in_ema","Age_s",f"label_will_cutoff_{K}s"]
    return df[cols]

# ---------------- user interface ----------------

def select_file(files):
    print("\nAvailable telemetry files:\n")
    for i,f in enumerate(files,1):
        print(f"{i}. {f.name}")
    while True:
        try:
            c=int(input("\nEnter number of file to process: "))
            if 1<=c<=len(files):
                return files[c-1]
        except ValueError:
            pass
        print("Invalid input. Try again.")

def main():
    folder_input = input("Enter folder path to scan (press Enter for current folder): ").strip()
    folder = Path(folder_input) if folder_input else Path.cwd()

    if not folder.exists():
        print("❌ Folder not found:", folder)
        return

    telem_files = [Path(f) for f in glob.glob(str(folder / "*telemetry*.csv"))]
    if not telem_files:
        print("❌ No telemetry CSVs found in", folder)
        return

    chosen = select_file(telem_files)
    print(f"\nUsing {chosen}\n")

    df = load_telemetry(chosen)
    cutoff = build_cutoff_table(df)
    cutoff_path = chosen.with_name(chosen.stem + "_cutoff_events.csv")
    cutoff.to_csv(cutoff_path, index=False)

    feats = build_features(df, cutoff, 15)
    feat_path = chosen.with_name(chosen.stem + "_risk_features_K15s.csv")
    feats.to_csv(feat_path, index=False)

    print(f"✅ Created:\n  {cutoff_path.name}\n  {feat_path.name}")

if __name__=="__main__":
    main()
