#!/usr/bin/env python3
"""batch.py SPEC.json --set DEV [--ref CHAMP-1.0.0] [--procs 11] - create (if new), run and compare a list of candidates.
SPEC: [{"id":..., "src":..., "config":{...}, "title":..., "hypothesis":..., "parent":...}, ...]. Existing candidates are
re-used only if their registered main.py and config match the spec exactly (never overwritten)."""
import os, sys, json, argparse, subprocess, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
sys.path.insert(0, HERE); import fbsopt as H   # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("--set", required=True); ap.add_argument("--ref", default="CHAMP-1.0.0"); ap.add_argument("--procs", type=int, default=11)
    a = ap.parse_args(); spec = json.load(open(a.spec)); summ = []
    for c in spec:
        d = H.cdir(c["id"]); src = os.path.abspath(os.path.join(os.path.dirname(a.spec), c["src"])) if not os.path.isabs(c["src"]) else c["src"]
        if os.path.exists(d):
            meta = json.load(open(os.path.join(d, "meta.json")))
            if meta["main_sha256"] != H.sha256f(src) or meta["config"] != c.get("config", {}): sys.exit("%s exists with a different main.py or config" % c["id"])
        else:
            subprocess.run([PY, os.path.join(HERE, "fbsopt.py"), "new", c["id"], "--src", src, "--config", json.dumps(c.get("config", {})), "--title", c["title"],
                            "--hypothesis", c["hypothesis"], "--parent", c.get("parent", "CHAMP-1.0.0")], check=True)
        if not os.path.exists(os.path.join(d, "results_%s.json" % a.set)):
            subprocess.run([PY, os.path.join(HERE, "fbsopt.py"), "run", c["id"], "--set", a.set, "--procs", str(a.procs)], check=True, stdout=subprocess.DEVNULL)
        if not os.path.exists(os.path.join(H.cdir(a.ref), "results_%s.json" % a.set)):
            subprocess.run([PY, os.path.join(HERE, "fbsopt.py"), "run", a.ref, "--set", a.set, "--procs", str(a.procs)], check=True, stdout=subprocess.DEVNULL)
        out = subprocess.run([PY, os.path.join(HERE, "fbsopt.py"), "compare", c["id"], "--set", a.set, "--ref", a.ref], check=True, capture_output=True, text=True).stdout
        rep = json.load(open(os.path.join(d, "compare_%s_vs_%s.json" % (a.set, a.ref)))); G = rep["groups"]
        line = "%-34s ALL %+6.2f%% [%+6.2f,%+6.2f] | EuRoC %+6.2f%% [%+6.2f,%+6.2f] | TUM %+6.2f%% [%+6.2f,%+6.2f] | ate %+5.1f/%+5.1f | ident %d fail %d dec %d | ms %.2f->%.2f" % (
            c["id"], G["ALL"]["composite"]["rel"], *G["ALL"]["composite"]["ci"], G["EuRoC"]["composite"]["rel"], *G["EuRoC"]["composite"]["ci"], G["TUM-VI"]["composite"]["rel"], *G["TUM-VI"]["composite"]["ci"],
            G["EuRoC"]["ate"]["rel"], G["TUM-VI"]["ate"]["rel"], rep["byte_identical"], len(rep["new_failures"]), len(rep["decision_changes"]), G["ALL"]["ms_per_imu_s"]["ref"], G["ALL"]["ms_per_imu_s"]["cand"])
        print(line, flush=True); summ.append(line)
    open(os.path.join(os.path.dirname(os.path.abspath(a.spec)), os.path.basename(a.spec).replace(".json", "_%s_summary.txt" % a.set)), "w").write("\n".join(summ) + "\n")

if __name__ == "__main__":
    main()
