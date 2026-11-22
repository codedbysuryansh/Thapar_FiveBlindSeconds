"""Batch J: learned-model retest (X060). The snapshot main_dev_J.py = main_dev_I.py + edits_12 (velocity aid from side files of
learned body-velocity predictions; research only, never shipped: the official contract forbids side files and the network
needs PyTorch). DEV0-dev-identity-J checks that the snapshot with all flags at their defaults is byte-identical to the
champion. X060a/b use HELD-OUT predictions (each mission predicted by the fold that excluded its environment: MH / V1 / V2 for
EuRoC, the mission's room for TUM-VI). X060c/d use IN-POOL predictions (the network saw the test missions in training): they
measure the memorisation threat only and can never be candidates."""
import json
snap = "../src/snapshots/main_dev_J.py"; specs = []
RUNS = "~/SamsungPRISM_FiveBlindSeconds/opt/learned/runs/X108b-final24-safe-2e-3-0.04__"
def add(cid, setd, title, hyp, parent="X108b-final24-safe-2e-3-0.04"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15],
         tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04)
H = "Learned 1-D CNN body-velocity predictions (heteroscedastic) fused as velocity rows reduce drift where the physics has no ZUPT"
add("X060a-velaid-loeo-s1", {"CFG": dict(B, vel_aid_dir=RUNS + "loeo", vel_aid_scale=1.0)}, "Velocity aid, held-out predictions (LOEO/LORO), predicted sigma x1", H)
add("X060b-velaid-loeo-s2", {"CFG": dict(B, vel_aid_dir=RUNS + "loeo", vel_aid_scale=2.0)}, "Velocity aid, held-out predictions (LOEO/LORO), predicted sigma x2", H)
add("X060c-velaid-inpool-s1", {"CFG": dict(B, vel_aid_dir=RUNS + "inpool", vel_aid_scale=1.0)}, "THREAT ONLY: velocity aid, in-pool predictions (test missions in training), sigma x1", H)
add("X060d-velaid-inpool-s2", {"CFG": dict(B, vel_aid_dir=RUNS + "inpool", vel_aid_scale=2.0)}, "THREAT ONLY: velocity aid, in-pool predictions (test missions in training), sigma x2", H)
json.dump(specs, open("batch_J.json", "w"), indent=1)
ident = [dict(id="DEV0-dev-identity-J", src=snap, config={}, title="Snapshot J with defaults must equal the champion", hypothesis="identity", parent="CHAMP-1.0.0")]
json.dump(ident, open("batch_J0.json", "w"), indent=1); print(len(specs), len(ident))
# J2: sigma-scale sweep to bound the best achievable effect (held-out: does any scale help? in-pool: the memorisation optimum)
specs2 = []
for cid, fold, sc in [("X060e-velaid-loeo-s4", "loeo", 4.0), ("X060f-velaid-loeo-s8", "loeo", 8.0), ("X060g-velaid-inpool-s4", "inpool", 4.0), ("X060h-velaid-inpool-s8", "inpool", 8.0)]:
    specs2.append(dict(id=cid, src=snap, config={"set": {"CFG": dict(B, vel_aid_dir=RUNS + fold, vel_aid_scale=sc)}},
                       title=("THREAT ONLY: " if fold == "inpool" else "") + "Velocity aid, %s predictions, predicted sigma x%g" % ("in-pool" if fold == "inpool" else "held-out (LOEO/LORO)", sc),
                       hypothesis=H, parent="X108b-final24-safe-2e-3-0.04"))
json.dump(specs2, open("batch_J2.json", "w"), indent=1); print(len(specs2))
