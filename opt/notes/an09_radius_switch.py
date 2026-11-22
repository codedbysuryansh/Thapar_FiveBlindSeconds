"""an09: would a switch on the loose solution's own radius (est_radius of X042f = what the rotor smoother would report as rr)
pick the tight drag prior where it helps? Offline mixing of per-mission results of X042f (loose) and X042c / X042d (tight)."""
import sys, numpy as np
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness"); import fbsopt as H
L = H.load_results("X042f-rest-dragoff-loose-F", "DEV"); C = H.load_results("CHAMP-1.0.0", "DEV")
for tight_id in ("X042c-rest-dragoff-p16b", "X042d-rest-dragoff-p16a", "X042e-rest-dragoff-sc0.1-tau1000"):
    T = H.load_results(tight_id, "DEV"); tags = [t for t in L if L[t]["dataset"] == "EuRoC"]
    def comp(choice, sel=lambda t: True):
        ts = [t for t in tags if sel(t)]; tot = 0
        for k, w in (("wp", 30), ("b_abs", 20), ("drop", 10)):
            a = np.mean([choice(t)[t]["score"][k] for t in ts]); b = np.mean([C[t]["score"][k] for t in ts]); tot += w * (a / b - 1)
        ate = np.mean([choice(t)[t]["score"]["ate"] for t in ts]) / np.mean([C[t]["score"]["ate"] for t in ts]) - 1
        return 100 * tot / 60, 100 * ate
    print(tight_id)
    print("  always loose  composite %+.2f%% ate %+.2f%%" % comp(lambda t: L)); print("  always tight  composite %+.2f%% ate %+.2f%%" % comp(lambda t: T))
    for thr in (1.4, 1.5, 1.6, 1.7, 1.8, 2.0):
        ch = lambda t, thr=thr: T if L[t]["score"]["est_radius"] >= thr else L
        g = {e: comp(ch, lambda t, e=e: H.env_group(L[t]["name"]) == e) for e in ("MH", "V1", "V2")}; n_t = sum(L[t]["score"]["est_radius"] >= thr for t in tags)
        print("  switch r>=%.1f (tight on %2d/%d) composite %+.2f%% ate %+.2f%% | MH %+.1f V1 %+.1f V2 %+.1f" % (thr, n_t, len(tags), *comp(ch), g["MH"][0], g["V1"][0], g["V2"][0]))
    print("  radii of the loose solution per sequence: " + ", ".join("%s %.2f-%.2f" % (s[:5], min(L[t]["score"]["est_radius"] for t in tags if L[t]["name"] == s), max(L[t]["score"]["est_radius"] for t in tags if L[t]["name"] == s)) for s in H.EUROC))
