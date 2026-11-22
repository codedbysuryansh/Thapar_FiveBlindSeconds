#!/usr/bin/env python3
"""Acquire EuRoC MAV non-image members (IMU, ground truth, Leica/Vicon, sensor.yaml) from the official
ETH Research Collection deposit (DOI 10.3929/ethz-b-000690084). Research instrument, not product code.
The deposit nests deflated per-sequence zips inside three outer zips, so each inner zip is streamed,
inflated to a temporary file, CRC-checked, the non-image members extracted, and the temporary file removed."""
import os, sys, json, zlib, time, hashlib, zipfile, urllib.request
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zip_range as zr
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "euroc"))
B = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/%s/content"
OUTER = {"machine_hall.zip": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe", "vicon_room1.zip": "02ecda9a-298f-498b-970c-b7c44334d880", "vicon_room2.zip": "ea12bc01-3677-4b4c-853d-87c7870b8c44"}
def plan():
    cache = os.path.join(ROOT, "PLAN.json")
    if os.path.exists(cache): return json.load(open(cache))
    jobs = []
    for outer, uid in OUTER.items():
        rf = zr.Remote(B % uid)
        for e in zr.entries(rf):
            if e["name"].endswith(".zip"):
                jobs.append({"outer": outer, "url": B % uid, "entry": e, "data_off": zr.data_offset(rf, e)})
    jobs = sorted(jobs, key=lambda j: j["entry"]["csize"]); json.dump(jobs, open(cache, "w"), indent=1)
    return jobs
def fetch(job):
    e = job["entry"]; seq = os.path.basename(e["name"])[:-4]; out = os.path.join(ROOT, seq)
    prov_path = os.path.join(ROOT, "PROVENANCE_%s.json" % seq)
    if os.path.exists(prov_path): return seq, "skip"
    tmp = os.path.join(ROOT, "_tmp_%s.zip" % seq)
    for attempt in range(4):
        try:
            t0 = time.time(); d = zlib.decompressobj(-15); crc = 0; got = 0
            req = urllib.request.Request(job["url"], headers={**zr.UA, "Range": "bytes=%d-%d" % (job["data_off"], job["data_off"] + e["csize"] - 1)})
            with zr._open(req, timeout=300) as r, open(tmp, "wb") as f:
                while True:
                    c = r.read(1 << 20)
                    if not c: break
                    got += len(c); u = d.decompress(c); crc = zlib.crc32(u, crc); f.write(u)
                u = d.flush(); crc = zlib.crc32(u, crc); f.write(u)
            if got != e["csize"] or (crc & 0xFFFFFFFF) != e["crc"]: raise IOError("size/CRC mismatch got=%d" % got)
            members = []
            with zipfile.ZipFile(tmp) as z:
                for zi in z.infolist():
                    n = zi.filename
                    if zi.is_dir() or n.lower().endswith((".png", ".jpg", ".jpeg")) or "__MACOSX" in n or os.path.basename(n).startswith("."): continue
                    data = z.read(zi); p = os.path.join(out, n); os.makedirs(os.path.dirname(p), exist_ok=True)
                    open(p, "wb").write(data)
                    members.append({"member": n, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
                n_img = sum(1 for zi in z.infolist() if zi.filename.lower().endswith(".png"))
            os.remove(tmp)
            json.dump({"dataset": "EuRoC MAV", "doi": "10.3929/ethz-b-000690084", "licence": "In Copyright - Non-Commercial Use Permitted (rightsstatements.org InC-NC/1.0)",
                       "outer_archive": job["outer"], "outer_url": job["url"], "inner_member": e["name"], "inner_crc32": "%08x" % e["crc"], "inner_bytes": e["usize"],
                       "images_not_kept": n_img, "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1), "members": members},
                      open(prov_path, "w"), indent=1)
            return seq, "ok %.0fs %d members" % (time.time() - t0, len(members))
        except Exception as ex:
            err = repr(ex); time.sleep(10 * (attempt + 1))
    return seq, "FAILED " + err
if __name__ == "__main__":
    jobs = plan(); print("planned", [(os.path.basename(j["entry"]["name"]), j["entry"]["csize"]) for j in jobs], flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as p:
        for seq, status in p.imap_unordered(fetch, jobs): print(time.strftime("%H:%M:%S"), seq, status, flush=True)
    print("DONE", flush=True)
