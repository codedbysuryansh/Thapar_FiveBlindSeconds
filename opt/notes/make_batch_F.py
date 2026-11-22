"""Batch F: workspace-adaptive drag-bias prior (X043). The tight prior helps the large-hall flights (MH -9..-31 % ATE) and
hurts the small-room V1 flights (+2..+21 %); the guard's hand-held smoother radius rh (already computed on every rotor input,
no extra cost) measures the workspace size, so the prior is chosen from it. Physical reading: in a confined room the
drone's own recirculating downwash makes the apparent drag offset wander (loose prior); in a large hall it is steady."""
import json, shutil, hashlib
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_F.py"; shutil.copy2(SRC, snap)
specs = []
def add(cid, setd, title, hyp, parent="CHAMP-1.0.0"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-F", {}, "main_dev.py (batch F build) with every flag off", "Flags off must reproduce the champion byte for byte")
R = {"rest_frac": 0.5, "rest_gate": 0.06, "drag_off_rest": 1}
H = "Choose the drag-bias prior from the workspace size (guard radius rh): tight in large spaces, loose in confined rooms"
add("X042f-rest-dragoff-loose-F", {"CFG": dict(R)}, "Rest 50 %/0.06 + no drag rows at rest (records rh; = X042a on build F)", "Reference for the radius analysis")
for thr in (1.3, 1.5, 1.7, 2.0):
    add("X043-adapt-r%g-p16b" % thr, {"CFG": dict(R, drag_adapt_r=thr, drag_tight=[0.05, 300.0, 0.15])}, "Rest + no drag rows at rest + tight drag prior (0.15/0.05/300) when rh >= %g m" % thr, H)
for thr in (1.5, 1.7):
    add("X043-adapt-r%g-p16a" % thr, {"CFG": dict(R, drag_adapt_r=thr, drag_tight=[0.03, 1000.0, 0.15])}, "Rest + no drag rows at rest + tighter drag prior (0.15/0.03/1000) when rh >= %g m" % thr, H)
json.dump(specs, open("batch_F.json", "w"), indent=1); print(len(specs), hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
