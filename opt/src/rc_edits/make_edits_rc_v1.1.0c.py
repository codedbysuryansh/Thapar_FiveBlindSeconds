"""Writes edits_rc_v1.1.0c.json: RC v1.1.0b -> v1.1.0c = the clean implementation of X110d (guard checked on the loose rotor
solution; the steady re-smooth is attempted only if that guard passes and is kept only if it passes the same guard, otherwise
the loose solution is recomputed and used)."""
import json
E = []
E.append(["""        pk, vk = smooth_rotor(Mz); rr = radius(Mz.bt, pk, vk); bt = Mz.bt
        if rr >= CFG["switch_r"]:                                     # large workspace: steady drag offset, smoothed again
            del pk, vk; pk, vk = smooth_rotor(Mz, DRAG_STEADY); rr = radius(Mz.bt, pk, vk); steady = True
        _mark("smoother-rotor")
        if rr > CFG["guard"] * max(rh, 1e-6): guard = True; del pk, vk; Mz = Meas(D, F, att, CFG["dec_hand"], False, k_drag); D.A = None""",
"""        pk, vk = smooth_rotor(Mz); rr = radius(Mz.bt, pk, vk); bt = Mz.bt; lim = CFG["guard"] * max(rh, 1e-6)
        if CFG["switch_r"] <= rr <= lim:                              # large workspace: steady drag offset, smoothed again
            del pk, vk; pk, vk = smooth_rotor(Mz, DRAG_STEADY); steady = True
            if radius(Mz.bt, pk, vk) > lim:                           # the steady solution fails the guard: back to the loose one
                del pk, vk; pk, vk = smooth_rotor(Mz); steady = False; reverted = True
        _mark("smoother-rotor")
        if rr > lim: guard = True; del pk, vk; Mz = Meas(D, F, att, CFG["dec_hand"], False, k_drag); D.A = None"""])
E.append(["""_mark("measurements"); guard = False; steady = False; t = D.t""", """_mark("measurements"); guard = False; steady = False; reverted = False; t = D.t"""])
E.append(["""guard=guard, steady_drag=steady, long_gravity=long_g,""", """guard=guard, steady_drag=steady, steady_reverted=reverted, long_gravity=long_g,"""])
E.append(["""the ground reaction), gap-scaled accelerometer-bias variance; when the rotor solution spans a large workspace (radius >= 1.7 m)
it is smoothed again with a steady drag-row offset; radius guard; blind window: rotor = damped velocity (0.5 s), hand-held = the
smoother's own prediction, orientation held.""",
"""the ground reaction), gap-scaled accelerometer-bias variance; radius guard on the rotor solution (RMS radius above 12x that of
the hand-held smoother on the same rows -> hand-held solution); when the rotor solution passes it and spans a large workspace
(radius >= 1.7 m) it is smoothed again with a steady drag-row offset, kept only if it passes the same guard (else the first
rotor solution is used); blind window: rotor = damped velocity (0.5 s), hand-held = the smoother's own prediction, orientation
held."""])
json.dump(E, open("edits_rc_v1.1.0c.json", "w"), indent=1); print(len(E))
