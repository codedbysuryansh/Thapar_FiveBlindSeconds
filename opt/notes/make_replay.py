"""make_replay.py OUT.json ID [ID ...] - a batch spec that re-uses existing candidates (their own main.py and config) for
another mission set (replication / confirmation)."""
import json, sys, os
sys.path.insert(0, "../harness"); import fbsopt as H
out = sys.argv[1]; specs = []
for cid in sys.argv[2:]:
    meta = json.load(open(os.path.join(H.cdir(cid), "meta.json")))
    specs.append(dict(id=cid, src=os.path.join(H.cdir(cid), "main.py"), config=meta["config"], title=meta["title"], hypothesis=meta["hypothesis"], parent=meta.get("parent")))
json.dump(specs, open(out, "w"), indent=1); print(len(specs), "candidates ->", out)
