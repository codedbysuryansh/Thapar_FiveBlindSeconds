"""Batch M (round 3, accuracy fine-tuning of the two accepted mechanisms, interaction-aware): base = X110d (= X108b on every
official set). X116: the steady drag-offset prior of the switched branch (only flights with a loose radius >= 1.7 m use it, so
round 1's fixed-prior sweeps P15-P17, run for all flights and before the rest/drag changes, do not answer this). X117: the
hand-held Matern-5/2 prior re-tuned with the 24 s final gravity (round 1's P09 sweep used the 3 s gravity)."""
import json
snap = "../src/snapshots/main_dev_L.py"; specs = []
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_final_hand=24.0,
         final_mu_max=0.002, final_sw_max=0.04, guard_ref="revert")
def add(cid, cfg, extra, title, hyp):
    st = {"CFG": cfg}; st.update(extra); specs.append(dict(id=cid, src=snap, config={"set": st}, title=title, hypothesis=hyp, parent="X110d-guard-revert"))
HS = "The switched (large-workspace) branch has its own optimum, not measured in round 1"
for cid, dt in (("X116a-steady-sc0.02", [0.02, 1000.0, 0.15]), ("X116b-steady-sc0.04", [0.04, 1000.0, 0.15]), ("X116c-steady-tau3000", [0.03, 3000.0, 0.15]),
                ("X116d-steady-sc0_0.10", [0.03, 1000.0, 0.10]), ("X116e-steady-sc0_0.25", [0.03, 1000.0, 0.25]), ("X116f-steady-tau300", [0.03, 300.0, 0.15])):
    add(cid, dict(B, drag_tight=dt), {}, "Steady drag prior of the switched branch: sd %g m/s^2, tau %g s, initial sd %g" % tuple(dt), HS)
HM = "With less tilt leakage (24 s final gravity) the hand-held motion prior has a different optimum"
for cid, mt in (("X117a-matern-ell1.25", [1.25, 2.0, 20.0, 0.1]), ("X117b-matern-ell0.8", [0.8, 2.0, 20.0, 0.1]), ("X117c-matern-sb0.07", [1.0, 2.0, 20.0, 0.07]),
                ("X117d-matern-sb0.14", [1.0, 2.0, 20.0, 0.14]), ("X117e-matern-taub40", [1.0, 2.0, 40.0, 0.1]), ("X117f-matern-sp3", [1.0, 3.0, 20.0, 0.1])):
    add(cid, dict(B), {"MATERN": mt}, "Hand-held Matern prior ell %g s, sigma_p %g m, tau_b %g s, sigma_b %g (with the 24 s final gravity)" % tuple(mt), HM)
json.dump(specs, open("batch_M.json", "w"), indent=1); print(len(specs))
