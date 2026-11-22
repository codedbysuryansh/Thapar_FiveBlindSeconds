"""Writes edits_13_guard_switch.json: X110 (guard checked on the loose solution, before the drag switch), X111 (soft switch:
blend of the loose and steady solutions with weight clip((rr0 - lo) / (hi - lo), 0, 1)), X112 (switch only if the loose rotor
radius is at least `switch_ratio` times the hand-held guard radius). All defaults reproduce snapshot J exactly."""
import json
E = []
E.append(["           vel_aid_dir=None, vel_aid_scale=1.0, vel_aid_floor=0.05)  # X060 (research only): learned body-velocity predictions fused as velocity rows",
          "           vel_aid_dir=None, vel_aid_scale=1.0, vel_aid_floor=0.05,  # X060 (research only): learned body-velocity predictions fused as velocity rows\n"
          "           guard_ref=\"final\", switch_blend=None, switch_ratio=None)  # X110 guard on the loose radius (\"loose\"); X111 soft switch [lo, hi] m; X112 switch only if rr0 >= ratio * rh"])
E.append(["""        if CFG["drag_switch_r"] is not None and dsel is None and rr >= CFG["drag_switch_r"]:     # X044: large workspace -> steady drag offset
            del pk, vk; pk, vk = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); rr = radius(Mz.bt, pk, vk); RAD["tight"] = True
        RAD["rr"] = rr; _mark("smoother-rotor")
        if rr > CFG["guard"] * max(rh, 1e-6):""",
"""        loose_guard = CFG["guard_ref"] == "loose" and rr > CFG["guard"] * max(rh, 1e-6)         # X110: the guard checks the loose solution first
        ratio_ok = CFG["switch_ratio"] is None or rr >= CFG["switch_ratio"] * max(rh, 1e-6)       # X112: the drag model must explain most of the workspace
        if CFG["switch_blend"] is not None and dsel is None and not loose_guard and ratio_ok:       # X111: soft switch between the loose and the steady solution
            lo_, hi_ = CFG["switch_blend"]; wb = min(1.0, max(0.0, (rr - lo_) / (hi_ - lo_))); RAD["blend"] = wb
            if wb > 0.0:
                pk2, vk2 = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); RAD["tight"] = wb >= 1.0
                if wb >= 1.0: pk, vk = pk2, vk2
                else: pk = (1.0 - wb) * pk + wb * pk2; vk = (1.0 - wb) * vk + wb * vk2
                del pk2, vk2; rr = radius(Mz.bt, pk, vk)
        elif CFG["drag_switch_r"] is not None and dsel is None and not loose_guard and ratio_ok and rr >= CFG["drag_switch_r"]:     # X044: large workspace -> steady drag offset
            del pk, vk; pk, vk = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); rr = radius(Mz.bt, pk, vk); RAD["tight"] = True
        RAD["rr"] = rr; _mark("smoother-rotor")
        if loose_guard or (CFG["guard_ref"] != "loose" and rr > CFG["guard"] * max(rh, 1e-6)):"""])
json.dump(E, open("edits_13_guard_switch.json", "w"), indent=1); print(len(E))
