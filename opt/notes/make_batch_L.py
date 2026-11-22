"""Batch L: X110d (guard on the loose solution; a steady re-smooth that fails the same 12x guard falls back to the loose rotor
solution instead of the hand-held one). Snapshot L = K + edits_14."""
import json
snap = "../src/snapshots/main_dev_L.py"
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04)
specs = [dict(id="X110d-guard-revert", src=snap, config={"set": {"CFG": dict(B, guard_ref="revert")}}, parent="X108b-final24-safe-2e-3-0.04",
              title="X108b + guard on the loose solution; a steady solution failing the guard reverts to the loose rotor solution",
              hypothesis="Do-no-harm fallback: T4 showed the steady solution is sometimes bad when it fails the 12x guard (MH_02 stress cases) and sometimes very good (MH_04); reverting to the loose rotor solution avoids the hand-held fallback losses of X108b and the steady losses of X110a")]
json.dump(specs, open("batch_L.json", "w"), indent=1)
json.dump([dict(id="DEV0-dev-identity-L", src=snap, config={}, title="Snapshot L with defaults must equal the champion", hypothesis="identity", parent="CHAMP-1.0.0")], open("batch_L0.json", "w"), indent=1)
