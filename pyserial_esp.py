#!/usr/bin/env python3
"""
pyserial_esp.py



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
- Periodically sends: "ot state", "ot parent", "ot counters mac"
  (prefix "xxxx:...:0:ff:fe00:" + parentRloc16).

Debugging:
- Prints the first 50 serial lines it sees: [reader] ...
- Prints when it parses state, parent, counters, switches, BetterParent, mesh prefix.

Usage example:

    python pyserial_esp.py --port COM8 --baud 115200 --duration 300 --interval 1.0
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

DEFAULT_PORT = "COM8"       # change if needed
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

    # keep track of last time we saw parent / counters
    last_parent_ts: Optional[dt.datetime] = None
    last_counters_ts: Optional[dt.datetime] = None

    # RLOC prefix string in EXACT PS-style form:
    #   "fdde:ad00:beef:0:0:ff:fe00:"
    mesh_local_prefix: Optional[str] = None


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

# RLOC IPv6 address, just like PS "Get-MyRlocAddr":
#   ^[0-9a-f:]+:0:ff:fe00:[0-9a-f]{1,4}$
RLOC_ADDR_RE = re.compile(
    r"\b([0-9a-fA-F:]+:0:ff:fe00:[0-9a-fA-F]{1,4})\b",
    re.IGNORECASE,
)

STATES = {"child", "router", "leader", "detached", "disabled"}

# Strip ANSI/VT100 escape sequences
ANSI_ESCAPE_RE = re.compile(r'\x1B\[[0-9;?]*[ -/]*[@-~]')


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_parent_rloc_ipv6(prefix_ps_style: str, rloc16: str) -> str:
    """Build parent RLOC IPv6 EXACTLY like the PS script."""
    rloc16 = (rloc16 or "").strip()
    try:
        r = int(rloc16, 16)
        r_hex = f"{r:04x}"
    except ValueError:
        r_hex = rloc16.lower()
    return (prefix_ps_style + r_hex).lower()


def process_line(ts_iso: str,
                 line: str,
                 tstate: TelemetryState,
                 parent_events_writer: csv.writer):
    """
    Process a single serial line:
      - log has already been written
      - update telemetry state
      - write parent_events rows when relevant
      - DEBUG prints when state/parent/counters/etc. are detected
    """
    stripped = line.strip()

    # --- learn RLOC prefix EXACTLY like PS Get-MyRlocAddr + regex ---
    if tstate.mesh_local_prefix is None:
        m_addr = RLOC_ADDR_RE.search(line)
        if m_addr:
            my_rloc = m_addr.group(1).lower()
            prefix = re.sub(
                r'([0-9a-f:]+:0:ff:fe00:)[0-9a-f]{1,4}$',
                r'\1',
                my_rloc,
            )
            if prefix and prefix != my_rloc:
                with tstate.lock:
                    if tstate.mesh_local_prefix is None:
                        tstate.mesh_local_prefix = prefix
                        print(f"[parse] Mesh-local prefix (PS-style) learned: {prefix}")

        # --- child/router/leader/detached/disabled from "state"
    st_lower = stripped.lower()
    if st_lower in STATES:
        with tstate.lock:
            # store original text (for CSV) but also act on the logical state
            tstate.last_state = stripped
            if st_lower in ("detached", "disabled"):
                # ChildTelemetry.ps1 semantics:
                # when we are detached/disabled, there is NO valid parent anymore.
                tstate.parent_rloc16 = None
                tstate.parent_extaddr = None
                tstate.lqi_in = None
                tstate.lqi_out = None
                tstate.age_s = None
        print(f"[parse] STATE -> {stripped}")


    # --- parent ext addr
    m = PARENT_EXT_RE.search(line)
    if m:
        ext = m.group(1).lower()
        with tstate.lock:
            tstate.parent_extaddr = ext
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)
        print(f"[parse] Parent ExtAddr -> {ext}")

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
        print(f"[parse] Parent RLOC16 -> {r_hex}")

    # --- LQI in/out
    m = LQI_IN_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.lqi_in = val
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)
        print(f"[parse] LQI_IN -> {val}")

    m = LQI_OUT_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.lqi_out = val
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)
        print(f"[parse] LQI_OUT -> {val}")

    # --- Age
    m = AGE_RE.search(stripped)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.age_s = val
            tstate.last_parent_ts = dt.datetime.now(dt.timezone.utc)
        print(f"[parse] Age -> {val}s")

    # --- MAC counters
    now = dt.datetime.now(dt.timezone.utc)

    m = TX_TOTAL_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.tx_total = val
            tstate.last_counters_ts = now
        print(f"[parse] TxTotal -> {val}")

    m = RX_TOTAL_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.rx_total = val
            tstate.last_counters_ts = now
        print(f"[parse] RxTotal -> {val}")

    m = TX_ERR_CCA_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.tx_err_cca = val
            tstate.last_counters_ts = now
        print(f"[parse] TxErrCca -> {val}")

    m = TX_RETRY_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.tx_retry = val
            tstate.last_counters_ts = now
        print(f"[parse] TxRetry -> {val}")

    m = RX_ERR_FCS_RE.search(line)
    if m:
        val = int(m.group(1))
        with tstate.lock:
            tstate.rx_err_fcs = val
            tstate.last_counters_ts = now
        print(f"[parse] RxErrFcs -> {val}")

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
        print(f"[parse] PARENT SWITCH {from_rloc} -> {to_rloc}")

        # also update current parent in telemetry
        with tstate.lock:
            tstate.parent_rloc16 = to_rloc

    # --- Better parent attempts
    if "BetterParent" in line:
        parent_events_writer.writerow([
            ts_iso, "better_parent", "", "", line
        ])
        print(f"[parse] BetterParent event")

    # --- Optional: NoAck failures as events too
    if "error:NoAck" in line:
        parent_events_writer.writerow([
            ts_iso, "noack", "", "", line
        ])
        print(f"[parse] NoAck error event")


# ------------ Reader thread ------------

def reader_thread_fn(ser: serial.Serial,
                     stop_event: threading.Event,
                     tstate: TelemetryState,
                     raw_log_file,
                     parent_events_writer: csv.writer):
    """
    Continuously read lines from serial, timestamp them, log them, and parse.
    Debug: prints first ~50 lines it sees.
    """
    debug_count = 0

    while not stop_event.is_set():
        try:
            line_bytes = ser.readline()
        except serial.SerialException as e:
            print(f"[reader] SerialException: {e}", file=sys.stderr)
            break

        if not line_bytes:
            continue

        try:
            line = line_bytes.decode("utf-8", errors="replace")
            line = ANSI_ESCAPE_RE.sub("", line)
            line = line.rstrip("\r\n")
        except Exception:
            continue

        ts_iso = iso_now()
        raw_log_file.write(f"{ts_iso} {line}\n")
        raw_log_file.flush()

        # DEBUG: show first N lines
        if debug_count < 50:
            print(f"[reader] {ts_iso} | {line}")
            debug_count += 1

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
    raw_log_file.write(f"[INFO] esp logger started at {iso_now()}\n")
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

        # -------- set TX power + log level 
        try:
            print("[INFO] Setting txpower -10 dBm on OpenThread CLI")
            ser.write(b"txpower -20\r\n")
            time.sleep(0.1)

            print("[INFO] Setting OpenThread log level 5")
            ser.write(b"log level 5\r\n")
            time.sleep(0.1)
        except serial.SerialException as e:
            print(f"[main] Failed to configure txpower/log level: {e}", file=sys.stderr)

        # -------- DISCOVER RLOC ADDRESS (ipaddr) --------
        try:
            print("[INFO] Requesting ipaddr to learn mesh-local prefix")
            ser.write(b"ipaddr\r\n")
            time.sleep(0.2)
        except serial.SerialException as e:
            print(f"[main] Failed to send ipaddr: {e}", file=sys.stderr)

        # -------- TELEMETRY LOOP --------
        print(f"[INFO] Starting telemetry loop for {duration_s} s, interval {interval_s} s")
        t_start = time.time()
        t_end = t_start + duration_s

        while time.time() < t_end:
            now = time.time()

            # build ping command to parent if we know prefix + RLOC16
            with tstate.lock:
                mesh_prefix = tstate.mesh_local_prefix
                parent_rloc16 = tstate.parent_rloc16

            ping_cmd = None
            if mesh_prefix and parent_rloc16:
                parent_ip = build_parent_rloc_ipv6(mesh_prefix, parent_rloc16)
                ping_cmd = f"ping {parent_ip}\r\n".encode("ascii")
                print(f"[main] Will ping parent {parent_ip}")
            else:
                print(f"[main] No parent_ip yet (mesh_prefix={mesh_prefix}, parent_rloc16={parent_rloc16})")

            # Send commands – reader thread will see and parse their output
            try:
                ser.write(b"state\r\n")
                ser.write(b"parent\r\n")
                ser.write(b"counters mac\r\n")
                if ping_cmd is not None:
                    ser.write(ping_cmd)
            except serial.SerialException as e:
                print(f"[main] Serial write failed: {e}", file=sys.stderr)
                break

            time.sleep(0.05)  # tiny delay to let some responses arrive

                        # snapshot telemetry
            now_iso = iso_now()
            now_utc = dt.datetime.now(dt.timezone.utc)

            with tstate.lock:
                last_state = tstate.last_state
                parent_rloc16 = tstate.parent_rloc16
                parent_extaddr = tstate.parent_extaddr
                lqi_in = tstate.lqi_in
                lqi_out = tstate.lqi_out
                age_s = tstate.age_s
                tx_total = tstate.tx_total
                rx_total = tstate.rx_total
                tx_err_cca = tstate.tx_err_cca
                tx_retry = tstate.tx_retry
                rx_err_fcs = tstate.rx_err_fcs
                last_parent_ts = tstate.last_parent_ts

            # --- "truthful" parent semantics like ChildTelemetry.ps1 ---
            # 1) If we're not a child/router, there is effectively no parent.
            parent_rloc16_out = parent_rloc16
            parent_extaddr_out = parent_extaddr

            st_lower = (last_state or "").lower()
            if st_lower in ("detached", "disabled", ""):
                parent_rloc16_out = None
                parent_extaddr_out = None
            else:
                # 2) If we haven't seen any parent info recently, treat as no parent
                parent_stale = (
                    last_parent_ts is None
                    or (now_utc - last_parent_ts).total_seconds() > max(2.5 * interval_s, 3.0)
                )
                if parent_stale:
                    parent_rloc16_out = None
                    parent_extaddr_out = None

            row = [
                now_iso,
                last_state,
                parent_rloc16_out,
                parent_extaddr_out,
                lqi_in,
                lqi_out,
                age_s,
                tx_total,
                rx_total,
                tx_err_cca,
                tx_retry,
                rx_err_fcs,
            ]

            telem_writer.writerow(row)
            telem_file.flush()
            print(f"[main] Snapshot row: {row}")


            telem_writer.writerow(row)
            telem_file.flush()
            print(f"[main] Snapshot row: {row}")

            # wait until next interval
            sleep_remaining = interval_s - (time.time() - now)
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

        print("[INFO] Telemetry loop finished.")

    finally:
        # shutdown
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
    parser = argparse.ArgumentParser(description="esp nRF52 OpenThread serial logger")
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Serial port (default {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD,
                        help="Baud rate (default 115200)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="Telemetry interval in seconds")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                        help="Total duration in seconds")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP,
                        help="Warmup seconds before telemetry")

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
