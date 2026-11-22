"""Batch K (round 2, robustness of RC v1.1.0b). T4 on rc v1.1.0b found 24 radius-guard flips on Machine Hall flights (the
steady drag prior enlarges the rotor radius ~1.5-1.8x, so the unchanged 12x guard fires on sound solutions, ATE +10 %), switch
misfires under strong non-official perturbations, and hand-held losses of the 24 s final gravity under some gyro conditions.
X110: the guard checks the loose solution (its calibrated quantity) before the switch; X110b/c: shorter final hand-held gravity
(12 s / 8 s, same clean-gyro conditions); X111: soft switch (blend weight linear in the loose radius) instead of the hard 1.7 m
threshold, which sits inside the MH_02 radius range (1.52-1.87 m: decisions change between seeds).
(X112, a switch gated on rr0/rh, was dropped before running: on DEV/CONF the MH_03 flights switch at rr0/rh 1.75-1.9 with
-20..-42 % ATE, inside the range of the T4 misfires 1.7-3.1, so no ratio threshold separates them.)"""
import json
snap = "../src/snapshots/main_dev_K.py"; specs = []
def add(cid, setd, title, hyp, parent="X108b-final24-safe-2e-3-0.04"):
    specs.append(dict(id=cid, src=snap, config={"set": setd} if setd else {}, title=title, hypothesis=hyp, parent=parent))
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15],
         tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04)
G = dict(B, guard_ref="loose")
add("X110a-guardloose", {"CFG": G}, "X108b + radius guard on the loose solution before the drag switch",
    "The guard threshold (12x) was calibrated on loose rotor solutions; checking it there removes the guard flips without changing any official-data decision")
add("X110b-guardloose-final12", {"CFG": dict(G, tau_g_final_hand=12.0)}, "X110a + 12 s final hand-held gravity (clean-gyro conditions unchanged)",
    "A 12 s final gravity keeps the DEV TUM-VI gain of 24 s (X108d -7.98 % vs -7.95 %) and integrates gyro faults over half the time")
add("X110c-guardloose-final8", {"CFG": dict(G, tau_g_final_hand=8.0)}, "X110a + 8 s final hand-held gravity (clean-gyro conditions unchanged)",
    "An 8 s final gravity is still more robust to gyro faults at a small accuracy cost")
add("X111a-blend-1.5-2.0", {"CFG": dict(G, switch_blend=[1.5, 2.0])}, "X110a + soft switch: weight of the steady solution = clip((rr0 - 1.5) / 0.5, 0, 1)",
    "A soft switch removes the knife-edge at 1.7 m (MH_02) and halves the damage of marginal misfires")
add("X111b-blend-1.6-1.9", {"CFG": dict(G, switch_blend=[1.6, 1.9])}, "X110a + soft switch: weight = clip((rr0 - 1.6) / 0.3, 0, 1)",
    "A narrow soft switch around 1.7 m keeps the hard-switch decisions away from the edge")
json.dump(specs, open("batch_K.json", "w"), indent=1)
json.dump([dict(id="DEV0-dev-identity-K", src=snap, config={}, title="Snapshot K with defaults must equal the champion", hypothesis="identity", parent="CHAMP-1.0.0")],
          open("batch_K0.json", "w"), indent=1)
print(len(specs))
