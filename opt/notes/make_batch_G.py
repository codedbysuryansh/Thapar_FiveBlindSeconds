"""Batch G: radius-switched drag prior (X044), heteroscedastic accelerometer noise (X005), rotor blind damping time (X030),
and the combinations built from the screened directions."""
import json, shutil, hashlib
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_G.py"; shutil.copy2(SRC, snap)
specs = []
def add(cid, setd, title, hyp, parent="CHAMP-1.0.0"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-G", {}, "main_dev.py (batch G build) with every flag off", "Flags off must reproduce the champion byte for byte")
R = {"rest_frac": 0.5, "rest_gate": 0.06, "drag_off_rest": 1}
HS = "an09: re-smoothing with the tight drag prior only when the loose rotor solution's radius is >= ~1.7 m keeps the MH gain (-18.7 %) without the V1 loss (+0.5 % instead of +8.9 %)"
for thr in (1.6, 1.7, 1.8):
    add("X044-switch-r%g-p16a" % thr, {"CFG": dict(R, drag_switch_r=thr, drag_tight=[0.03, 1000.0, 0.15])}, "Rest + no drag rows at rest + re-smooth with drag prior 0.15/0.03/1000 when radius >= %g m" % thr, HS)
add("X044-switch-r1.7-p16b", {"CFG": dict(R, drag_switch_r=1.7, drag_tight=[0.05, 300.0, 0.15])}, "Rest + no drag rows at rest + re-smooth with drag prior 0.15/0.05/300 when radius >= 1.7 m", HS)
HH = "The degradation's white noise grows with time (sev^1.7 ramp); a per-chunk accelerometer variance from the data instead of one global value weights early and late rows correctly"
add("X005a-hetacc", {"CFG": {"het_acc": 1}}, "Heteroscedastic accelerometer rows (per-chunk high-frequency variance, clipped 0.2-5 x global)", HH)
add("X005b-hetacc-narrow", {"CFG": {"het_acc": 1, "het_lo": 0.5, "het_hi": 2.0}}, "Heteroscedastic accelerometer rows, clipped 0.5-2 x global", HH)
HB = "Rotor blind window: damped velocity with time constant 0.5 s; with better velocity estimates a longer or shorter constant may be better"
for tb in (0.3, 0.8, 1.2):
    add("X030-blindtau-%g" % tb, {"CFG": {"blind_tau_r": tb}}, "Rotor blind-window damping time %g s (champion 0.5)" % tb, HB)
HC = "Combination of the screened directions: rotor rest/ground handling + radius-switched drag prior; hand-held long gravity time constant (+ acceleration compensation)"
add("X104-combo5", {"CFG": dict(R, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_hand=24.0)}, "Combination 5: rest + no drag at rest + switched drag prior (r 1.7, 0.15/0.03/1000) + hand-held tau_g 24", HC)
add("X105-combo6", {"CFG": dict(R, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_hand=24.0, acc_comp=1)}, "Combination 6: combination 5 + one acceleration-compensated gravity pass (hand-held)", HC)
add("X106-combo7", {"CFG": dict(R, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_hand=24.0, het_acc=1)}, "Combination 7: combination 5 + heteroscedastic accelerometer rows", HC)
json.dump(specs, open("batch_G.json", "w"), indent=1); print(len(specs), hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
