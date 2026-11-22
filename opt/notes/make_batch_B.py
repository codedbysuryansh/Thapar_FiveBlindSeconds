"""Batch B: first structural candidates on main_dev.py (rest/stillness detection, drag rows off at rest, acceleration-compensated gravity)."""
import json, shutil, hashlib, os
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_B.py"; os.makedirs("../src/snapshots", exist_ok=True); shutil.copy2(SRC, snap)
specs = []
def add(cid, cfg, title, hyp, parent="CHAMP-1.0.0"):
    specs.append(dict(id=cid, src=snap, config={"set": {"CFG": cfg}} if cfg else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-B", {}, "main_dev.py (batch B build) with every flag off", "Flags off must reproduce the champion byte for byte")
H41 = "The rest detector admits only the quietest 5 % of 0.5 s windows; EuRoC flights sit still on the ground for 7-24 s (an01) and drift there without zero-velocity rows"
add("X041a-rest_frac-0.15", dict(rest_frac=0.15), "Rest detector: candidate windows up to 15 %", H41)
add("X041b-rest_frac-0.30", dict(rest_frac=0.30), "Rest detector: candidate windows up to 30 %", H41)
add("X041c-rest_frac-0.30-gate-0.05", dict(rest_frac=0.30, rest_gate=0.05), "Rest detector: 30 %, gyro gate 0.05 rad/s", H41)
add("X041d-rest_frac-0.50-gate-0.05", dict(rest_frac=0.50, rest_gate=0.05), "Rest detector: 50 %, gyro gate 0.05 rad/s", H41)
add("X041e-rest30-g05-dragoff", dict(rest_frac=0.30, rest_gate=0.05, drag_off_rest=1), "Rest detector 30 % / 0.05 and no drag rows at rest", H41 + "; drag rows on the ground (rotors spinning) measure ground tilt, not velocity")
add("X041f-rest15-dragoff", dict(rest_frac=0.15, drag_off_rest=1), "Rest detector 15 % and no drag rows at rest", H41)
H01 = "Motion acceleration leaks into the 3 s gravity smoother (hand-held circular motion has periods of a few seconds); subtracting the smoother's own acceleration estimate before re-smoothing gravity reduces the tilt error (the -16 % tilt oracle of E56)"
add("X001a-acccomp-1-hand", dict(acc_comp=1), "Acceleration-compensated gravity, one iteration, hand-held only", H01)
add("X001b-acccomp-2-hand", dict(acc_comp=2), "Acceleration-compensated gravity, two iterations, hand-held only", H01)
add("X001c-acccomp-1-both", dict(acc_comp=1, acc_comp_rotor=1), "Acceleration-compensated gravity, one iteration, both platforms", H01)
add("X001d-acccomp-1-hand-g05", dict(acc_comp=1, acc_comp_gain=0.5), "Acceleration-compensated gravity, one iteration at half gain, hand-held", H01)
json.dump(specs, open("batch_B.json", "w"), indent=1); print(len(specs), "candidates; snapshot sha256", hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
