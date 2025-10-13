# ppk2_reader.py — minimal, Windows-friendly
import argparse, csv, time, datetime, sys
try:
    from ppk2_api.ppk2_api import PPK2
except Exception as e:
    print("ERROR: ppk2-api not installed. Run: py -m pip install ppk2-api", file=sys.stderr)
    raise

def iso_now():
    return datetime.datetime.now().isoformat(timespec="milliseconds")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--hz", type=int, default=20, help="samples per second")
    ap.add_argument("--seconds", type=int, default=300, help="duration")
    args = ap.parse_args()

    ppk = PPK2()  # first PPK2 found
    try:
        ppk.use_ampere_meter()  # ammeter mode (we are not sourcing)
    except Exception:
        pass

    period = 1.0 / max(1, args.hz)
    t_end = time.time() + args.seconds

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","current_mA"])
        time.sleep(0.1)
        while time.time() < t_end:
            t0 = time.time()
            # preferred: average since last call
            current_A = None
            try:
                current_A = ppk.get_next_data_point()
            except Exception:
                current_A = 0.0
            w.writerow([iso_now(), round((current_A or 0.0) * 1000.0, 6)])  # A→mA
            rem = period - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)

    try: ppk.stop_measuring()
    except: pass
    try: ppk.close()
    except: pass

if __name__ == "__main__":
    main()
