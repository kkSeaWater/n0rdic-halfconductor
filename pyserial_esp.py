#!/usr/bin/env python3
"""
esp_logger.py

Robust serial logger for ESP32-H2 OpenThread CLI.

- Continuously reads all serial lines (no data loss).
- Logs raw lines with timestamps to esp_log_*.txt
- Extracts:
    - state             (child/router/leader/detached/disabled)
    - parent RLOC16
    - parent ExtAddr
    - LQI in/out
    - Age
    - MAC counters: TxTotal, RxTotal, TxErrCca, TxRetry, RxErrFcs
    - Parent switches: "RLOC16 xxxx -> yyyy"
    - BetterParent attempts
- Periodically sends: "state", "parent", "counters mac"

Usage example:
    python esp_logger.py --port COM4 --baud 115200 --duration 300 --interval 1.0
"""

import argparse
import datetime as dt
import threading
import time
import re
import sys
import csv
from dataclasses import dataclass, field
from typing import Optional

import serial  # pip install pyserial


# ------------ Config / defaults ------------

DEFAULT_PORT = "COM4"
DEFAULT_BAUD = 115200
DEFAULT_INTERVAL = 1.0       # seconds between telemetry snapshots
DEFAULT_DURATION = 300.0     # total seconds of experiment
DEFAULT_WARMUP = 2.0         # warmup seconds before starting commands


# ------------ Telemetry state ------------

@dataclass
class TelemetryState:
    lock: threading.Lock = field(default_factory=threading.Lock)

    last_state: Optional[str] = None

    parent_rloc16: Optional[str] = None
    parent_extaddr: Optional[str] = None
    lqi_in: Optional[int] = None
    lqi_out: Optional[int] = None
    age_s: Optional[int] = None

    tx_total: Optional[int] = None
    rx_total: Optional[int] = None
    tx_err_cca: Optional[int] = None
    tx_retry: Optional[int] = None
    rx_err_fcs: Optional[int] = None

    # keep track of last time we saw parent / counters for sanity if needed
    last_parent_ts: Optional[dt.datetime] = None
    last_counters_ts: Optional[dt.datetime] = None


# ------------ Line parsing helpers ------------

RLOC_SWITCH_RE = re.compile(r"RLOC16\s+([0-9a-fA-F]{1,4})\s*->\s*([0-9a-fA-F]{1,4})")
PARENT_EXT_RE = re.compile(r"Ext Addr:\s*([0-9a-fA-F]{16})", re.IGNORECASE)
PARENT_RLOC_RE = re.compile(r"Rloc:\s*([0-9a-fA-F]{1,4})", re.IGNORECASE)
LQI_IN_RE = re.compile(r"Link\s*Quality\s*In:\s*(\d+)", re.IGNORECASE)
LQI_OUT_RE = re.compile(r"Link\s*Quality\s*Out:\s*(\d+)", re.IGNORECASE)
AGE_RE = re.compile(r"^Age:\s*(\d+)")
TX_TOTAL_RE = re.compile(r"TxTotal:\s*(\d+)")
RX_TOTAL_RE = re.compile(r"RxTotal:\s*(\d+)")
TX_ERR_CCA_RE = re.compile(r"TxErrCca:\s*(\d+)", re.IGNORECASE)
TX_RETRY_RE = re.compile(r"TxRetry:\s*(\d+)", re.IGNORECASE)
RX_ERR_FCS_RE = re.compile(r"RxErrFcs:\s*(\d+)", re.IGNORECASE)

STATES = {"child", "router", "leader", "detached", "disabled"}

# --- PATCH: strip ANSI escape sequences (the “Japanese” crap) ---
# This matches VT100/ANSI sequences like ESC[0K, ESC[6n, ESC[3C etc.
ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-9;?]*[ -/]*[@-~]')


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process_line(ts_iso: str,
                 line: str,
                 tstate: TelemetryState,
                 parent_events_writer: csv.writer):
    """
    Process a single serial line:
      - write to raw log (done outside)
      - update telemetry state
      - write parent_events rows when relevant
    """
    stripped = line.strip()

    # --- child/router/leader/detached/disabled from "state"
    if stripped.lower() in STATES:
        with tstate.lock:
            tstate.last_state = stripped
        # fall-through: keep parsing other patterns in same line if any

    # --- parent ext addr
    m = PARENT_EXT_RE.search(line)
    if m:
        ext = m.group(1).lower()
        with tstate.lock:
            tstate.parent_extaddr = ext
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)

    # --- parent RLOC hex
    m = PARENT_RLOC_RE.search(line)
    if m:
        r = m.group(1)
        try:
            r_hex = f"{int(r, 16):04x}"
        except ValueError:
            r_hex = r.lower()
        with tstate.lock:
            tstate.parent_rloc16 = r_hex
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)

    # --- LQI in/out
    m = LQI_IN_RE.search(line)
    if m:
        with tstate.lock:
            tstate.lqi_in = int(m.group(1))
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)

    m = LQI_OUT_RE.search(line)
    if m:
        with tstate.lock:
            tstate.lqi_out = int(m.group(1))
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)

    # --- Age
    m = AGE_RE.search(line)
    if m:
        with tstate.lock:
            tstate.age_s = int(m.group(1))
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)

    # --- MAC counters
    now = dt.datetime.now(dt.timezone.utc)

    m = TX_TOTAL_RE.search(line)
    if m:
        with tstate.lock:
            tstate.tx_total = int(m.group(1))
            tstate.last_counters_ts = now

    m = RX_TOTAL_RE.search(line)
    if m:
        with tstate.lock:
            tstate.rx_total = int(m.group(1))
            tstate.last_counters_ts = now

    m = TX_ERR_CCA_RE.search(line)
    if m:
        with tstate.lock:
            tstate.tx_err_cca = int(m.group(1))
            tstate.last_counters_ts = now

    m = TX_RETRY_RE.search(line)
    if m:
        with tstate.lock:
            tstate.tx_retry = int(m.group(1))
            tstate.last_counters_ts = now

    m = RX_ERR_FCS_RE.search(line)
    if m:
        with tstate.lock:
            tstate.rx_err_fcs = int(m.group(1))
            tstate.last_counters_ts = now

    # --- Parent switch events "RLOC16 xxxx -> yyyy"
    m = RLOC_SWITCH_RE.search(line)
    if m:
        from_rloc_raw, to_rloc_raw = m.group(1), m.group(2)
        try:
            from_rloc = f"{int(from_rloc_raw, 16):04x}"
        except ValueError:
            from_rloc = from_rloc_raw.lower()
        try:
            to_rloc = f"{int(to_rloc_raw, 16):04x}"
        except ValueError:
            to_rloc = to_rloc_raw.lower()

        parent_events_writer.writerow([
            ts_iso, "parent_switch", from_rloc, to_rloc, line
        ])

        # also update current parent in telemetry
        with tstate.lock:
            tstate.parent_rloc16 = to_rloc

    # --- Better parent attempts
    if "BetterParent" in line:
        parent_events_writer.writerow([
            ts_iso, "better_parent", "", "", line
        ])

    # --- Optional: NoAck failures as events too
    if "error:NoAck" in line:
        parent_events_writer.writerow([
            ts_iso, "noack", "", "", line
        ])


# ------------ Reader thread ------------

def reader_thread_fn(ser: serial.Serial,
                     stop_event: threading.Event,
                     tstate: TelemetryState,
                     raw_log_file,
                     parent_events_writer: csv.writer):
    """
    Continuously read lines from serial, timestamp them, log them, and parse.
    """
    while not stop_event.is_set():
        try:
            line_bytes = ser.readline()
        except serial.SerialException as e:
            print(f"[reader] SerialException: {e}", file=sys.stderr)
            break

        if not line_bytes:
            # timeout; just loop again
            continue

        try:
            line = line_bytes.decode("utf-8", errors="replace")
            # --- PATCH: strip ANSI before anything else ---
            line = ANSI_ESCAPE_RE.sub("", line)
            line = line.rstrip("\r\n")
        except Exception:
            continue

        ts_iso = iso_now()
        # raw log with timestamp (now CLEAN, no escape codes)
        raw_log_file.write(f"{ts_iso} {line}\n")
        raw_log_file.flush()

        # parse line for telemetry & events
        process_line(ts_iso, line, tstate, parent_events_writer)


# ------------ Main telemetry loop ------------

def run_logger(port: str,
               baud: int,
               interval_s: float,
               duration_s: float,
               warmup_s: float):
    # timestamp base for filenames
    ts_base = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_txt = f"esp_log_{ts_base}.txt"
    telemetry_csv = f"esp_telemetry_{ts_base}.csv"
    parent_events_csv = f"esp_parent_events_{ts_base}.csv"

    print(f"[INFO] Opening serial {port} @ {baud} baud")
    ser = serial.Serial(port, baudrate=baud, timeout=0.1)

    # open files
    raw_log_file = open(log_txt, "w", encoding="utf-8", newline="\n")
    telem_file = open(telemetry_csv, "w", encoding="utf-8", newline="")
    events_file = open(parent_events_csv, "w", encoding="utf-8", newline="")

    telem_writer = csv.writer(telem_file)
    events_writer = csv.writer(events_file)

    # write headers
    raw_log_file.write(f"[INFO] ESP logger started at {iso_now()}\n")
    telem_writer.writerow([
        "timestamp_iso",
        "state",
        "parent_rloc16",
        "parent_extaddr",
        "lqi_in",
        "lqi_out",
        "age_s",
        "tx_total",
        "rx_total",
        "tx_err_cca",
        "tx_retry",
        "rx_err_fcs",
    ])
    events_writer.writerow([
        "timestamp_iso",
        "event_type",
        "from_rloc16",
        "to_rloc16",
        "raw_line"
    ])

    tstate = TelemetryState()
    stop_event = threading.Event()

    # start reader thread
    reader_thread = threading.Thread(
        target=reader_thread_fn,
        args=(ser, stop_event, tstate, raw_log_file, events_writer),
        daemon=True
    )
    reader_thread.start()

    try:
        # -------- WARMUP --------
        print(f"[INFO] Warmup for {warmup_s} s...")
        warmup_end = time.time() + warmup_s
        while time.time() < warmup_end:
            time.sleep(0.1)

        # -------- NEW: set TX power to 0 dBm --------
        try:
            print("[INFO] Setting txpower -10 dBm on OpenThread CLI")
            ser.write(b"txpower -10\r\n")
            # (optional) tiny delay to let it respond
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"[main] Failed to set txpower: {e}", file=sys.stderr)

        # -------- TELEMETRY LOOP --------
        print(f"[INFO] Starting telemetry loop for {duration_s} s, interval {interval_s} s")
        t_start = time.time()
        t_end = t_start + duration_s

        while time.time() < t_end:
            now = time.time()

            # Send commands – reader thread will see and parse their output
            try:
                ser.write(b"state\r\n")
                ser.write(b"parent\r\n")
                ser.write(b"counters mac\r\n")
            except serial.SerialException as e:
                print(f"[main] Serial write failed: {e}", file=sys.stderr)
                break

            time.sleep(0.05)  # tiny delay to let some responses arrive

            # snapshot telemetry (unchanged)
            with tstate.lock:
                row = [
                    iso_now(),
                    tstate.last_state,
                    tstate.parent_rloc16,
                    tstate.parent_extaddr,
                    tstate.lqi_in,
                    tstate.lqi_out,
                    tstate.age_s,
                    tstate.tx_total,
                    tstate.rx_total,
                    tstate.tx_err_cca,
                    tstate.tx_retry,
                    tstate.rx_err_fcs,
                ]

            telem_writer.writerow(row)
            telem_file.flush()

            # wait until next interval
            sleep_remaining = interval_s - (time.time() - now)
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

        print("[INFO] Telemetry loop finished.")

    finally:
        # shutdown (unchanged)
        stop_event.set()
        try:
            ser.close()
        except Exception:
            pass

        reader_thread.join(timeout=2.0)

        raw_log_file.close()
        telem_file.close()
        events_file.close()

        print(f"[INFO] Logs written:\n  {log_txt}\n  {telemetry_csv}\n  {parent_events_csv}")


# ------------ CLI ------------

def main():
    parser = argparse.ArgumentParser(description="ESP32-H2 OpenThread serial logger")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"Serial port (default {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Baud rate (default {DEFAULT_BAUD})")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Telemetry interval in seconds")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Total duration in seconds")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP, help="Warmup seconds before telemetry")

    args = parser.parse_args()

    run_logger(
        port=args.port,
        baud=args.baud,
        interval_s=args.interval,
        duration_s=args.duration,
        warmup_s=args.warmup,
    )


if __name__ == "__main__":
    main()
