"""Batch D: drag-bias prior grid with a separate initial SD (SC0), wider rest gates, longer hand-held gravity time constants,
and the first combinations - all on main_dev.py snapshot C (same build as batch C; flags off verified byte-identical)."""
import json, hashlib
snap = "../src/snapshots/main_dev_C.py"
specs = []
def add(cid, setd, title, hyp, parent="CHAMP-1.0.0"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
HD = "Drag-row bias = a sensor offset (initial SD SC0) plus a slow drift (OU with stationary SD SC, time constant TAU_C); batch C: smaller drift and a wide initial prior both help EuRoC"
for sc0 in (0.10, 0.20, 0.30):
    add("P17-sc0_%g-sc0.03-tauc1000" % sc0, {"SC0": sc0, "SC": 0.03, "TAU_C": 1000.0}, "Drag bias: initial sd %g, drift sd 0.03 / tau 1000 s" % sc0, HD)
for sc, tc in ((0.02, 1000.0), (0.01, 1000.0), (0.05, 3000.0), (0.03, 300.0), (0.02, 3000.0)):
    add("P17-sc0_0.15-sc%g-tauc%g" % (sc, tc), {"SC0": 0.15, "SC": sc, "TAU_C": tc}, "Drag bias: initial sd 0.15, drift sd %g / tau %g s" % (sc, tc), HD)
for sc, tc in ((0.02, 1000.0), (0.03, 1000.0), (0.02, 300.0)):
    add("P17-sc%g-tauc%g" % (sc, tc), {"SC": sc, "TAU_C": tc}, "Drag bias: sd %g, tau %g s (no separate initial sd)" % (sc, tc), HD)
HR = "Rotor-on ground windows of V1/V2 have gyro sd 0.025-0.034 rad/s and flight windows 0.10-0.28: a wider gate admits more true rest (batch C: gate 0.06 -2.3 % on EuRoC)"
for g in (0.08, 0.10):
    add("X041i-rest50-g%g" % g, {"CFG": {"rest_frac": 0.5, "rest_gate": g}}, "Rest detector 50 %%, gyro gate %g rad/s" % g, HR)
add("X041j-rest80-g06", {"CFG": {"rest_frac": 0.8, "rest_gate": 0.06}}, "Rest detector 80 %, gyro gate 0.06 rad/s", HR)
HG = "Hand-held tilt improves monotonically with a longer gravity time constant up to 12 s (batch C); find where it turns"
for tg in (16.0, 24.0, 40.0):
    add("X002-taug_hand-%g" % tg, {"CFG": {"tau_g_hand": tg}}, "Gravity time constant %g s on the hand-held path" % tg, HG)
for tg in (12.0, 24.0):
    add("X003-taug_hand-%g-acccomp1" % tg, {"CFG": {"tau_g_hand": tg, "acc_comp": 1}}, "Hand-held: gravity tau %g s + one acceleration-compensated pass" % tg, HG)
HT = "With the tighter drag prior the rotor attitude may also prefer a different gravity time constant"
for tg in (4.5, 6.0):
    add("X004-taug_rotor-%g-dragP16" % tg, {"CFG": {"tau_g_rotor": tg}, "SC0": 0.15, "SC": 0.05, "TAU_C": 300.0}, "Rotor gravity tau %g s with the drag prior of P16 (0.15 / 0.05 / 300)" % tg, HT)
HC = "Combination: the best of each family (to be re-selected by leave-one-environment-out before confirmation)"
add("X101-combo2", {"CFG": {"rest_frac": 0.5, "rest_gate": 0.06, "tau_g_hand": 12.0}, "SC0": 0.15, "SC": 0.03, "TAU_C": 1000.0}, "Combination 2: rest 50 %/0.06 + drag bias 0.15/0.03/1000 + hand-held tau_g 12", HC)
add("X102-combo3", {"CFG": {"rest_frac": 0.5, "rest_gate": 0.06, "tau_g_hand": 12.0, "acc_comp": 1}, "SC0": 0.15, "SC": 0.03, "TAU_C": 1000.0}, "Combination 3: combination 2 + one acceleration-compensated gravity pass (hand-held)", HC)
add("X103-combo4", {"CFG": {"rest_frac": 0.5, "rest_gate": 0.06, "tau_g_hand": 12.0}, "SC": 0.04, "TAU_C": 300.0}, "Combination 4: rest 50 %/0.06 + drag bias 0.04/300 + hand-held tau_g 12", HC)
json.dump(specs, open("batch_D.json", "w"), indent=1); print(len(specs), "candidates")
