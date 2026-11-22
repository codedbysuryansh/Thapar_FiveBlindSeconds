#!/usr/bin/env python3
"""Output-contract check (Phase 4 test T1), independent of main.py.

    python tools/check_output.py INPUT OUTPUT [--json]

INPUT is what main.py was given (zip, folder or CSV), OUTPUT the trajectory it wrote. The input is read here with the
standard library only. Checks, per file:
  rows        one output line per input data row (lines that are blank or start with '#' or a letter are not data rows)
  format      exactly 8 fields separated by single spaces, LF line ends, no blank lines (one leading '#' header allowed)
  timestamps  field 1 equals "%d.%09d" of the input nanoseconds, in input order (non-integer input stamps: within 1 us)
  decimals    positions with 6 and quaternion components with 7 decimals, all finite
  unit        | |q| - 1 | < 1e-6 for every row
Exit code 0 when every check passes."""
import sys, os, json, math, zipfile, io

def _member(names):
    names = [n for n in names if not n.startswith("__MACOSX") and not n.endswith("/")]; low = [n.replace("\\", "/").lower() for n in names]
    for want in ("imu0/data.csv", "data.csv"):
        hits = sorted((len(l), n) for n, l in zip(names, low) if l == want or l.endswith("/" + want))
        if hits: return hits[0][1]
    csv = [n for n, l in zip(names, low) if l.endswith(".csv")]
    return csv[0] if len(csv) == 1 else (sorted(csv, key=len)[0] if csv else None)
def read_input_stamps(path):
    """Timestamp field of every data row of the IMU CSV inside path, as strings, in order."""
    if os.path.isdir(path):
        files = [os.path.relpath(os.path.join(r, f), path) for r, _d, fs in os.walk(path) for f in fs]
        m = _member([f for f in files if f.lower().endswith(".csv")])
        if m is None: path = os.path.join(path, sorted([f for f in files if f.lower().endswith(".zip")], key=len)[0])
        else: path = os.path.join(path, m)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z: raw = z.read(_member(z.namelist()))
    else:
        with open(path, "rb") as f: raw = f.read()
    out = []
    for ln in raw.decode("utf-8", "replace").split("\n"):
        s = ln.strip()
        if not s or s[0] == "#" or s[0].isalpha(): continue
        out.append((s.split(",")[0] if "," in s else s.split()[0]).strip())
    return out
def expected_stamp(tok, scale_ns):
    """The output timestamp string for an input token (integer stamps -> exact; others numerically)."""
    try: v = int(tok); return "%d.%09d" % divmod(v * scale_ns, 10 ** 9), None
    except ValueError: return None, float(tok)
def unit_scale(stamps):
    """ns per input unit, as main.py infers it (median step below 20 us in ns -> the stamps are us, ms or s)."""
    ints = []
    for s_ in stamps[:20000]:
        try: ints.append(int(s_))
        except ValueError: return 1
    d = sorted(b - a for a, b in zip(ints, ints[1:]) if b > a)
    if not d: return 1
    med = d[len(d) // 2] if len(d) % 2 else 0.5 * (d[len(d) // 2 - 1] + d[len(d) // 2])
    if med < 2e4:
        for sc in (10 ** 3, 10 ** 6, 10 ** 9):
            if 1e9 / (med * sc) <= 5000: return sc
    return 1
def check(inp, outp):
    rep = dict(input=inp, output=outp, ok=False, problems=[])
    stamps = read_input_stamps(inp); rep["rows_in"] = len(stamps); sc = unit_scale(stamps)
    with open(outp, "rb") as f: raw = f.read()
    rep["bytes"] = len(raw); P = rep["problems"]
    if b"\r" in raw: P.append("CR characters in the output")
    if raw and not raw.endswith(b"\n"): P.append("last line not terminated by LF")
    lines = raw.decode("ascii", "replace").split("\n")
    if lines and lines[-1] == "": lines = lines[:-1]
    if lines and lines[0].startswith("#"): lines = lines[1:]; rep["header"] = True
    rep["rows_out"] = len(lines)
    if len(lines) != len(stamps): P.append("row count %d != input rows %d" % (len(lines), len(stamps)))
    bad_fmt = bad_ts = bad_dec = bad_fin = bad_unit = 0; worst_unit = 0.0; first = {}
    for i, ln in enumerate(lines):
        f = ln.split(" ")
        if len(f) != 8 or any(x == "" for x in f):
            bad_fmt += 1; first.setdefault("format", (i, ln)); continue
        if i < len(stamps):
            want, num = expected_stamp(stamps[i], sc)
            if want is not None and f[0] != want: bad_ts += 1; first.setdefault("timestamp", (i, f[0], want))
            if want is None and abs(float(f[0]) - num) > 1e-6: bad_ts += 1; first.setdefault("timestamp", (i, f[0], num))
        dec = [len(x.split(".")[1]) if "." in x else 0 for x in f[1:]]
        if dec != [6, 6, 6, 7, 7, 7, 7]: bad_dec += 1; first.setdefault("decimals", (i, ln))
        try: v = [float(x) for x in f[1:]]
        except ValueError: bad_fin += 1; first.setdefault("finite", (i, ln)); continue
        if not all(math.isfinite(x) for x in v): bad_fin += 1; first.setdefault("finite", (i, ln)); continue
        e = abs(math.sqrt(sum(x * x for x in v[3:])) - 1.0); worst_unit = max(worst_unit, e)
        if e >= 1e-6: bad_unit += 1; first.setdefault("unit", (i, ln))
    rep.update(bad_format=bad_fmt, bad_timestamp=bad_ts, bad_decimals=bad_dec, non_finite=bad_fin, bad_unit_quaternion=bad_unit, max_unit_error=worst_unit, first_failures=first)
    for k in ("bad_format", "bad_timestamp", "bad_decimals", "non_finite", "bad_unit_quaternion"):
        if rep[k]: P.append("%s: %d rows" % (k, rep[k]))
    rep["ok"] = not P; return rep
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2: print(__doc__); sys.exit(2)
    r = check(args[0], args[1])
    if "--json" in sys.argv: print(json.dumps(r, indent=1))
    else: print("%s  rows in %d, out %d  max | |q|-1 | %.1e  %s" % ("PASS" if r["ok"] else "FAIL", r["rows_in"], r.get("rows_out", -1), r.get("max_unit_error", -1), "; ".join(r["problems"])))
    sys.exit(0 if r["ok"] else 1)
