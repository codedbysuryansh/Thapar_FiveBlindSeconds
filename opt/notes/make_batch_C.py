"""Batch C: refinement of the batch A/B signals - drag-bias prior grid, platform-specific gravity time constant, rest gate, and
first combinations (main_dev.py snapshot C; flags off = champion, verified again by DEV0-dev-identity-C)."""
import json, shutil, hashlib, os
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_C.py"; shutil.copy2(SRC, snap)
specs = []
def add(cid, setd, title, hyp, parent="CHAMP-1.0.0"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-C", {}, "main_dev.py (batch C build) with every flag off", "Flags off must reproduce the champion byte for byte")
HC = "The drag-row bias (OU, tau 100 s, sd 0.15 m/s^2 = 0.8 m/s of velocity at k 0.19) is loose enough to absorb real velocity; batch A: sd 0.08 -4.4 %, tau 300 -2.3 % on EuRoC"
for sc in (0.04, 0.06, 0.08, 0.10):
    for tc in (100.0, 300.0, 1000.0):
        if (sc, tc) == (0.08, 100.0): continue                      # = P08c
        add("P15-sc%g-tauc%g" % (sc, tc), {"SC": sc, "TAU_C": tc}, "Drag-bias prior sd %g m/s^2, tau %g s" % (sc, tc), HC)
for sc, tc in ((0.05, 300.0), (0.05, 1000.0), (0.03, 1000.0)):
    add("P16-sc0_0.15-sc%g-tauc%g" % (sc, tc), {"SC0": 0.15, "SC": sc, "TAU_C": tc}, "Drag bias: initial sd 0.15, stationary sd %g, tau %g s" % (sc, tc), HC + "; the initial offset (sensor bias) and the drift (degradation random walk) need different priors")
HG = "Hand-held tilt is limited by motion acceleration leaking into the 3 s gravity smoother (batch A: TAU_G 4.5/6 = -6.3/-7.9 % on TUM-VI, EuRoC neutral); make it platform-specific"
for tg in (4.5, 6.0, 8.0, 12.0):
    add("X002-taug_hand-%g" % tg, {"CFG": {"tau_g_hand": tg}}, "Gravity time constant %g s on the hand-held path only" % tg, HG)
for tg in (6.0, 8.0):
    add("X003-taug_hand-%g-acccomp1" % tg, {"CFG": {"tau_g_hand": tg, "acc_comp": 1}}, "Hand-held: gravity tau %g s + one acceleration-compensated gravity pass" % tg, HG + "; X001 removes the leaked acceleration explicitly, so the two may add up or overlap")
add("X041g-rest50-g06", {"CFG": {"rest_frac": 0.5, "rest_gate": 0.06}}, "Rest detector 50 %, gyro gate 0.06 rad/s", "Wider gyro gate admits the rotor-on ground windows of V1/V2 (still gyro sd 0.025-0.034)")
add("X041h-rest50-g05-minw4", {"CFG": {"rest_frac": 0.5, "rest_gate": 0.05, "rest_minw": 4}}, "Rest detector 50 %, gate 0.05, at least 4 windows (2 s)", "Longer minimum rest cluster reduces false rest in slow hover")
add("X100-combo1", {"CFG": {"rest_frac": 0.5, "rest_gate": 0.05, "tau_g_hand": 6.0, "acc_comp": 1}, "SC": 0.08}, "Combination 1: rest 50 %/0.05 + drag-bias sd 0.08 + hand-held tau_g 6 + acc-comp 1",
    "The accepted directions act on different parts (rest/drag on rotor, gravity on hand-held) and should add up")
json.dump(specs, open("batch_C.json", "w"), indent=1); print(len(specs), "candidates; snapshot", hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
