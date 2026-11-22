#!/usr/bin/env python3
"""patch.py FILE EDITS.json - apply exact, single-occurrence string replacements [[old, new], ...] to FILE (fails if any old
string is missing or ambiguous; nothing is written in that case)."""
import sys, json
f, e = sys.argv[1], sys.argv[2]; src = open(f).read(); edits = json.load(open(e))
for old, new in edits:
    c = src.count(old)
    if c != 1: sys.exit("edit not applied (%d occurrences): %r" % (c, old[:120]))
    src = src.replace(old, new)
open(f, "w").write(src); print("applied %d edits to %s" % (len(edits), f))
