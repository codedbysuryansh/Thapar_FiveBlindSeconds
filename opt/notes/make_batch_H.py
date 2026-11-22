"""Batch H: robust hand-held gravity. T4 showed the 24 s hand-held gravity time constant (X104 / RC v1.1.0a) is fragile
outside the official degradation: E46 hand-held stress cases +93 % ATE (gyro-bias ramps and random walks +100..+380 %),
E55 short MH prefixes (classified hand-held) up to x7. Variants keep the rotor part (X044) and change only the hand-held gravity."""
import json, shutil, hashlib
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_H.py"; shutil.copy2(SRC, snap)
specs = []
def add(cid, setd, title, hyp, parent="X104-combo5"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-H", {}, "main_dev.py (batch H build) with every flag off", "Flags off must reproduce the champion byte for byte", parent="CHAMP-1.0.0")
R = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15])
H = "Keep the robust 3 s gravity for bias estimation and the drift decision; use the long time constant only where it is safe"
add("X107f-rotor-only", {"CFG": dict(R)}, "Rotor part of X104 only (rest + no drag at rest + switched drag prior); hand-held = champion", "Separates the rotor gain from the hand-held change")
add("X107a-final24-const", {"CFG": dict(R, tau_g_final_hand=24.0)}, "Hand-held: bias and mode from the 3 s front end; final attitude with 24 s gravity when the bias is constant", H)
add("X107b-final12-const", {"CFG": dict(R, tau_g_final_hand=12.0)}, "Hand-held: as X107a with a 12 s final gravity", H)
add("X107g-final8-const", {"CFG": dict(R, tau_g_final_hand=8.0)}, "Hand-held: as X107a with an 8 s final gravity", H)
add("X107e-final24-always", {"CFG": dict(R, tau_g_final_hand=24.0, final_const_only=0)}, "Hand-held: final attitude with 24 s gravity in both bias modes", H)
add("X107c-taug6", {"CFG": dict(R, tau_g_hand=6.0)}, "Hand-held: 6 s gravity everywhere on the hand-held path", "A moderate constant may keep most of the gain with less fragility")
add("X107d-acccomp", {"CFG": dict(R, acc_comp=1)}, "Hand-held: 3 s gravity + one acceleration-compensated gravity pass", "Removes leaked motion acceleration without slowing the tilt tracker")
json.dump(specs, open("batch_H.json", "w"), indent=1); print(len(specs), hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
