#!/usr/bin/env python3
"""Range-request tools for remote tar archives (TUM-VI). Research instrument, not product code.
Extracts only small members (IMU, mocap, calibration) from a remote .tar without downloading imagery.
  tar_range.py map  <url> [n]                       sample headers across the archive
  tar_range.py get  <url> <outdir> <tail_MB> <pattern>...   extract matching members found in the last N MB
  tar_range.py span <url> <outdir> <a> <b> <pattern>...     extract matching members found in byte span [a,b)
"""
import sys, os, urllib.request, hashlib, json, time
UA = {"User-Agent": "Mozilla/5.0 (research; range-extract of non-image members)"}
def size_of(url):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60) as r:
        return int(r.headers["Content-Length"]), r.geturl()
def rng(url, a, b):
    h = dict(UA); h["Range"] = "bytes=%d-%d" % (a, b)
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
                return r.read()
        except Exception as e:
            err = e; time.sleep(2 * (k + 1))
    raise err
def parse_header(b):
    if len(b) < 512 or b[257:262] != b"ustar": return None
    name = b[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
    prefix = b[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
    if prefix: name = prefix + "/" + name
    try: size = int(b[124:136].split(b"\0", 1)[0].strip() or b"0", 8)
    except ValueError: return None
    return name, size, chr(b[156]) if b[156] else "0"
def scan(buf, base):
    i = 0
    while i + 512 <= len(buf):
        h = parse_header(buf[i:i + 512])
        if h:
            name, size, tf = h
            yield name, size, tf, base + i + 512
            i += 512 + ((size + 511) // 512) * 512 if tf in ("0", "7") else 512
        else: i += 512
def cmd_map(url, n=12, chunk=2 * 2**20):
    total, final = size_of(url); print("size=%d" % total)
    for k in range(n + 1):
        a = min(max(0, (total * k // n) // 512 * 512), max(0, (total - chunk) // 512 * 512))
        hs = list(scan(rng(final, a, min(total, a + chunk) - 1), a))
        print("@%5.1f%% off=%d: %s" % (100 * a / total, a, "; ".join("%s(%d)" % (h[0], h[1]) for h in hs[:2])))
def extract_span(url, outdir, a, b, pats, total=None, final=None):
    if total is None: total, final = size_of(url)
    a = max(0, a // 512 * 512); b = min(total, b)
    buf = rng(final, a, b - 1); hs = list(scan(buf, a)); got = []
    other = [h for h in hs if not h[0].endswith(".png")]
    print("span [%d,%d) of %d: %d headers, %d non-png: %s" % (a, b, total, len(hs), len(other), [(h[0], h[1]) for h in other][:20]))
    for nm, sz, tf, off in hs:
        if tf == "0" and any(p in nm for p in pats):
            data = buf[off - a: off - a + sz] if off + sz <= a + len(buf) else rng(final, off, off + sz - 1)
            p = os.path.join(outdir, nm); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(data)
            got.append({"member": nm, "bytes": sz, "tar_data_offset": off, "sha256": hashlib.sha256(data).hexdigest()})
    prov = os.path.join(outdir, "PROVENANCE_" + os.path.basename(url) + ".json")
    old = json.load(open(prov))["members"] if os.path.exists(prov) else []
    names = {g["member"] for g in got}
    json.dump({"dataset": "TUM-VI", "source_url": url, "resolved_url": final, "archive_bytes": total,
               "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "members": [o for o in old if o["member"] not in names] + got}, open(prov, "w"), indent=1)
    print("saved", [(g["member"], g["bytes"]) for g in got])
if __name__ == "__main__":
    c = sys.argv[1]
    if c == "map": cmd_map(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 12)
    elif c == "get":
        total, final = size_of(sys.argv[2]); extract_span(sys.argv[2], sys.argv[3], total - int(float(sys.argv[4]) * 2**20), total, sys.argv[5:], total, final)
    elif c == "span": extract_span(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:])
