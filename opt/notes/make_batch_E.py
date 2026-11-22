"""Batch E: drag rows switched off where rest (ground contact) is detected, with the loose and tighter drag-bias priors.
an08 showed the mechanism behind the tight-prior gains on MH: with rotors spinning on the ground the drag rows measure the
ground-reaction tilt (an implied offset of -0.65 m/s^2 on MH_04), the loose bias learns it and needs ~20 s after take-off to
unlearn it (velocity error up to 1 m/s). Removing the ground rows should give the MH gain without tightening the prior (which
hurts V1)."""
import json, shutil, hashlib
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_E.py"; shutil.copy2(SRC, snap)
specs = []
def add(cid, setd, title, hyp, parent="CHAMP-1.0.0"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-E", {}, "main_dev.py (batch E build) with every flag off", "Flags off must reproduce the champion byte for byte")
H = "Drag rows are invalid during ground contact with rotors spinning; excluding them where rest is detected keeps the ground-reaction offset out of the drag-bias estimate (an08)"
R = {"rest_frac": 0.5, "rest_gate": 0.06, "drag_off_rest": 1}
add("X042a-rest-dragoff-loose", {"CFG": dict(R)}, "Rest 50 %/0.06, no drag rows at rest, champion drag-bias prior", H)
add("X042b-rest-dragoff-sc0.1-tau300", {"CFG": dict(R), "SC": 0.1, "TAU_C": 300.0}, "Rest 50 %/0.06, no drag rows at rest, drag bias 0.1/300", H)
add("X042c-rest-dragoff-p16b", {"CFG": dict(R), "SC0": 0.15, "SC": 0.05, "TAU_C": 300.0}, "Rest 50 %/0.06, no drag rows at rest, drag bias 0.15/0.05/300", H)
add("X042d-rest-dragoff-p16a", {"CFG": dict(R), "SC0": 0.15, "SC": 0.03, "TAU_C": 1000.0}, "Rest 50 %/0.06, no drag rows at rest, drag bias 0.15/0.03/1000", H)
add("X042e-rest-dragoff-sc0.1-tau1000", {"CFG": dict(R), "SC": 0.1, "TAU_C": 1000.0}, "Rest 50 %/0.06, no drag rows at rest, drag bias 0.1/1000", H)
json.dump(specs, open("batch_E.json", "w"), indent=1); print(len(specs), hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
