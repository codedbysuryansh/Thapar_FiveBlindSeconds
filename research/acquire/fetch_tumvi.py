#!/usr/bin/env python3
"""Acquire TUM-VI room1..6 (512_16, EuRoC export) non-image members via HTTP range requests. Research instrument.
Finds region boundaries between image directories by bisection on tar header names, then extracts small members."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tar_range as tr
OUT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tumvi"))
PATS = [".csv", ".yaml", ".txt"]
def label(final, total, off, chunk=1 << 20):
    off = max(0, min(off, total - chunk)) // 512 * 512
    hs = list(tr.scan(tr.rng(final, off, off + chunk - 1), off))
    labs = {os.path.dirname(h[0]) for h in hs if h[0].endswith(".png")}
    nonpng = [h for h in hs if not h[0].endswith(".png")]
    return labs, nonpng
def run(name):
    url = "https://cdn2.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-%s_512_16.tar" % name
    total, final = tr.size_of(url); print(name, "bytes", total, flush=True)
    tr.extract_span(url, OUT, 0, 3 << 20, PATS, total, final)
    tr.extract_span(url, OUT, total - (14 << 20), total, PATS, total, final)
    grid = [int(total * k / 10) for k in range(1, 10)]; labs = []
    for g in grid:
        l, _ = label(final, total, g); labs.append(next(iter(l)) if len(l) == 1 else None)
    for i in range(len(grid) - 1):
        if labs[i] != labs[i + 1]:
            lo, hi = grid[i], grid[i + 1]
            while hi - lo > (4 << 20):
                mid = (lo + hi) // 2; l, nonpng = label(final, total, mid)
                if nonpng or len(l) != 1: lo, hi = mid - (3 << 20), mid + (3 << 20); break
                if next(iter(l)) == labs[i]: lo = mid
                else: hi = mid
            print("boundary", labs[i], "->", labs[i + 1], "near", lo, hi, flush=True)
            tr.extract_span(url, OUT, lo - (9 << 20), hi + (9 << 20), PATS, total, final)
if __name__ == "__main__":
    for n in sys.argv[1:]: run(n)
