"""Batch O (diagnostic oracles, never promotable): does the per-flight drag coefficient explain why a tight drag-bias prior
helps MH and hurts V1? The oracle uses each flight's own ground-truth fit of k (Phase 2 E05) - ground truth on purpose."""
import json, sys
sys.path.insert(0, "../harness"); import fbsopt as H
KO = dict(H.K_FIT, default=0.18726)
snapC = "../src/snapshots/main_dev_C.py"; CH = "../../champion/v1.0.0/release/main.py"
specs = [dict(id="ORACLE-k-champ", src=CH, config={"k_drag": KO}, title="ORACLE: champion with each flight's own fitted k", hypothesis="How much the leave-one-out k costs the champion", parent="CHAMP-1.0.0"),
         dict(id="ORACLE-k-P16tight", src=snapC, config={"k_drag": KO, "set": {"SC0": 0.15, "SC": 0.03, "TAU_C": 1000.0}}, title="ORACLE: tight drag bias (0.15/0.03/1000) with each flight's own k",
              hypothesis="If the V1 losses of the tight drag bias come from the k mismatch, they vanish with the flight's own k", parent="P16-sc0_0.15-sc0.03-tauc1000"),
         dict(id="ORACLE-k-combo2", src=snapC, config={"k_drag": KO, "set": {"CFG": {"rest_frac": 0.5, "rest_gate": 0.06, "tau_g_hand": 12.0}, "SC0": 0.15, "SC": 0.03, "TAU_C": 1000.0}},
              title="ORACLE: combination 2 with each flight's own k", hypothesis="Upper bound of online k estimation on top of combination 2", parent="X101-combo2")]
json.dump(specs, open("batch_O.json", "w"), indent=1); print(len(specs))
