"""Batch A: one-at-a-time screens of the champion's prior / noise constants (config overrides on the champion source)."""
import json
CH = "../../champion/v1.0.0/release/main.py"
ROTOR = [20.0, 2.0, 0.3, 1.0, 20.0, 0.1]; MATERN = [1.0, 2.0, 20.0, 0.1]
def rot(i, v): r = list(ROTOR); r[i] = v; return r
def mat(i, v): m = list(MATERN); m[i] = v; return m
specs = []
def add(cid, setd, title, hyp):
    specs.append(dict(id=cid, src=CH, config={"set": setd}, title=title, hypothesis=hyp, parent="CHAMP-1.0.0"))
for s, v in zip("abcd", (0.8, 1.1, 1.8, 2.5)): add("P01%s-sd_drag-%g" % (s, v), {"SD_DRAG": v}, "Drag-row noise SD_DRAG = %g (champion 1.4)" % v, "The drag rows are mis-weighted; a different per-sample noise changes velocity shrinkage")
names = ["T_p", "tau_v", "tau_a", "sigma_a", "tau_b", "sigma_b"]
for i, vals, code in ((3, (0.6, 1.5, 2.2), "P02"), (0, (12.0, 30.0, 45.0), "P03"), (1, (1.0, 4.0), "P04"), (2, (0.15, 0.6), "P05"), (5, (0.05, 0.2), "P06"), (4, (10.0, 40.0), "P07")):
    for s, v in zip("abc", vals): add("%s%s-rotor_%s-%g" % (code, s, names[i], v), {"ROTOR": rot(i, v)}, "Rotor prior %s = %g (champion %g)" % (names[i], v, ROTOR[i]), "Re-tuning the rotor motion prior at V5P4")
for s, (k, v) in zip("abcd", (("TAU_C", 30.0), ("TAU_C", 300.0), ("SC", 0.08), ("SC", 0.3))): add("P08%s-%s-%g" % (s, k.lower(), v), {k: v}, "Drag-bias prior %s = %g" % (k, v), "The drag-row bias process is mis-specified")
mn = ["ell", "sigma_p", "tau_b", "sigma_b"]
for s, (i, v) in zip("abcdefgh", ((0, 0.6), (0, 1.5), (1, 1.0), (1, 3.0), (3, 0.05), (3, 0.2), (2, 10.0), (2, 40.0))):
    add("P09%s-matern_%s-%g" % (s, mn[i], v), {"MATERN": mat(i, v)}, "Hand-held Matern prior %s = %g (champion %g)" % (mn[i], v, MATERN[i]), "Re-tuning the hand-held motion prior at V5P4")
for s, v in zip("abc", (2.0, 4.5, 6.0)): add("P10%s-tau_g-%g" % (s, v), {"TAU_G": v}, "Gravity smoother TAU_G = %g s (champion 3)" % v, "Gravity tracking bandwidth trades bias-drift tracking against acceleration leakage")
for s, v in zip("ab", (0.15, 0.4)): add("P11%s-k2-%g" % (s, v), {"CFG": {"k2": v}}, "Gap rotation law k2 = %g (champion 0.25)" % v, "Gap rotation uncertainty mis-scaled")
for s, v in zip("ab", (0.25, 1.0)): add("P12%s-bias_jump-%g" % (s, v), {"BIAS_JUMP": v}, "Accelerometer-bias jump at gaps = %g (champion 0.5)" % v, "Gap-scaled bias variance mis-scaled")
for s, v in zip("ab", (0.01, 0.05)): add("P13%s-sz-%g" % (s, v), {"SZ": v}, "Zero-velocity row noise SZ = %g (champion 0.02)" % v, "ZUPT weight")
for s, v in zip("ab", (0.05, 0.3)): add("P14%s-lever-%g" % (s, v), {"LEVER_PRIOR": v}, "Lever-arm prior = %g m (champion 0.15)" % v, "Lever-arm prior width")
json.dump(specs, open("batch_A.json", "w"), indent=1); print(len(specs), "candidates")
