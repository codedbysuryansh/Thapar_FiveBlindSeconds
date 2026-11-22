"""Batch I: safety conditions for the long hand-held gravity (X046). X108a records the statistics (residual constant gyro bias
|mu| after the constant fit, gyro sample noise sw) with the X107a behaviour, to set thresholds that never fire on the official
degradation and catch the gyro-fault stress cases."""
import json, shutil, hashlib
SRC = "../src/main_dev.py"; snap = "../src/snapshots/main_dev_I.py"; shutil.copy2(SRC, snap)
specs = []
def add(cid, setd, title, hyp, parent="X107a-final24-const"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
add("DEV0-dev-identity-I", {}, "main_dev.py (batch I build) with every flag off", "Flags off must reproduce the champion byte for byte", parent="CHAMP-1.0.0")
R = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15])
add("X108a-final24-const-stats", {"CFG": dict(R, tau_g_final_hand=24.0)}, "= X107a on build I (records |mu| and the gyro sample noise)", "Statistics for the X046 safety thresholds")
json.dump(specs, open("batch_I.json", "w"), indent=1); print(len(specs), hashlib.sha256(open(snap, "rb").read()).hexdigest()[:16])
