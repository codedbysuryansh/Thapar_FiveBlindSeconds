"""Writes edits_14_revert.json: guard_ref = "revert" (X110d). The guard checks the loose solution (as X110a); if the steady
re-smooth then fails the same 12x check, the loose solution is recomputed and used (a refinement that fails the sanity check
falls back to the unrefined rotor solution, not to the hand-held model). Defaults unchanged (snapshot K behaviour)."""
import json
E = [["""        loose_guard = CFG["guard_ref"] == "loose" and rr > CFG["guard"] * max(rh, 1e-6)         # X110: the guard checks the loose solution first""",
      """        loose_guard = CFG["guard_ref"] in ("loose", "revert") and rr > CFG["guard"] * max(rh, 1e-6)   # X110: the guard checks the loose solution first"""],
     ["""            del pk, vk; pk, vk = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); rr = radius(Mz.bt, pk, vk); RAD["tight"] = True
        RAD["rr"] = rr; _mark("smoother-rotor")""",
      """            del pk, vk; pk, vk = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); rr = radius(Mz.bt, pk, vk); RAD["tight"] = True
            if CFG["guard_ref"] == "revert" and rr > CFG["guard"] * max(rh, 1e-6):             # X110d: the steady solution fails the guard -> loose solution again
                del pk, vk; pk, vk = smooth_rotor(Mz, drag=None); rr = radius(Mz.bt, pk, vk); RAD["tight"] = False; RAD["reverted"] = True
        RAD["rr"] = rr; _mark("smoother-rotor")"""],
     ["""        if loose_guard or (CFG["guard_ref"] != "loose" and rr > CFG["guard"] * max(rh, 1e-6)):""",
      """        if loose_guard or (CFG["guard_ref"] == "final" and rr > CFG["guard"] * max(rh, 1e-6)):"""]]
json.dump(E, open("edits_14_revert.json", "w"), indent=1); print(len(E))
