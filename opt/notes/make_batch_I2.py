"""Batch I2: X046 safety thresholds (from X108a statistics: official-degradation hand-held missions on DEV have |mu| <= 0.00163
rad/s and gyro sample noise <= 0.0295 rad/s; the T4 failures have |mu| 0.0027-0.011 or noise 0.055-0.117)."""
import json, hashlib
snap = "../src/snapshots/main_dev_I.py"; specs = []
def add(cid, setd, title, hyp, parent="X108a-final24-const-stats"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
R = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15])
H = "Use the long hand-held gravity only when the gyro is clean: constant bias mode, small residual constant bias |mu| and small gyro sample noise"
add("X108b-final24-safe-2e-3-0.04", {"CFG": dict(R, tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04)}, "Final 24 s hand-held gravity if const mode, |mu| <= 0.002 rad/s, gyro noise <= 0.04 rad/s", H)
add("X108c-final24-safe-1.5e-3-0.035", {"CFG": dict(R, tau_g_final_hand=24.0, final_mu_max=0.0015, final_sw_max=0.035)}, "Final 24 s hand-held gravity if const mode, |mu| <= 0.0015, gyro noise <= 0.035", H)
add("X108d-final12-safe-2e-3-0.04", {"CFG": dict(R, tau_g_final_hand=12.0, final_mu_max=0.002, final_sw_max=0.04)}, "Final 12 s hand-held gravity if const mode, |mu| <= 0.002, gyro noise <= 0.04", H)
json.dump(specs, open("batch_I2.json", "w"), indent=1); print(len(specs))
