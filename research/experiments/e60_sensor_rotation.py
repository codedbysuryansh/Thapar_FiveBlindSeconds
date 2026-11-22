#!/usr/bin/env python3
"""E60 - sensor-frame rotation invariance of the frozen v5: the official degraded stream (seed 42, orders A/B) is re-expressed in a
rotated sensor frame (w' = R w, a' = R a for every row; zero rows stay zero, repeated rows stay repeated) for three random fixed
rotations plus a 90-degree roll. An estimator that assumed a mounting (e.g. gravity on z) would change; v5 should not."""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f, blind5 as B
ROTS = {"none": np.eye(3), "roll90": Rot.from_euler("x", 90, degrees=True).as_matrix()}
for i, s in enumerate((11, 22, 33)): ROTS["rand%d" % i] = Rot.random(random_state=s).as_matrix()
def one(args):
    n, o = args; m = p4.official(n, o, seed=42); out = {}
    for k, R in ROTS.items():
        F = f.front_final(m.tn, m.w @ R.T, m.a @ R.T); r = f.back_final(F, k_drag=m.kd, blind=B.final_blind); sc = p4.score(m, r["p"], None); out[k] = dict(ate=sc["ate"], wp=sc["wp"], b_abs=sc["b_abs"], rotor=r["rotor"], mode=r["mode"], lock=float(F["lock"]))
    return dict(name=n, order=o, m=out)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(one, [(n, o) for o in ("A", "B") for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e60_sensor_rotation.json"), "w"))
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        rr = [r for r in res if sel(r["name"])]
        print("[%s] " % g + " | ".join("%s ATE %.3f (max |dATE| %.3f, gate flips %d, mode flips %d)" % (k, np.mean([r["m"][k]["ate"] for r in rr]), max(abs(r["m"][k]["ate"] - r["m"]["none"]["ate"]) for r in rr),
              sum(r["m"][k]["rotor"] != r["m"]["none"]["rotor"] for r in rr), sum(r["m"][k]["mode"] != r["m"]["none"]["mode"] for r in rr)) for k in ROTS))
