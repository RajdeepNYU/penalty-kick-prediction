"""The pixel crops were built from the original tracks, so on clips where the tracked person is not the kicker the model reads someone else. Rebuild the three crops of P4rkC (body_hm, body_tor, legs12) from a kicker overlay; only clips whose last box moved more than 0.3 h. Run with Q4_OVL set to the overlay. check = same code without an overlay reproduces saved crops.
usage: Q4_OVL=<abs>/cache/kfixW python cap.py fix_crops.py K | Q4_OVL=<abs>/cache/kidW python cap.py fix_crops.py D | python cap.py fix_crops.py check -> cache/cropsfix_{tag}.npz, logs/cropsfix_{tag}.jpg (public clips: old, new)"""
import os, sys, pickle, numpy as np, cv2
from common import *
import crops5, crops8

TAG = sys.argv[1]; prv = set(pm["clip"]); ix = {c: j for j, c in enumerate(CLIPS)}; C8 = np.load(f"{PC}/crops8.npy", mmap_mode="r"); C5 = np.load(f"{PC}/crops5.npy", mmap_mode="r"); V8 = list(crops8.VIEWS); V5 = list(crops5.VIEWS); FL = np.load(f"{PC}/zoom3_meta.npz"); IX = {c: j for j, c in enumerate(FL["clips"])}
assert all(IX[c] == ix[c] for c in CLIPS)


def three(c):
    _, r8 = crops8.crops(c); _, r5 = crops5.crops(c)
    if r8 is None or r5 is None: return None
    return r8[0], r5[0], np.stack([r8[1][V8.index("body_hm")], r8[1][V8.index("body_tor")], r5[1][V5.index("legs12")]])


if TAG == "check":
    for c in [gt["clip"][5], gt["clip"][500], pm["clip"][7]]:
        f8, f5, x = three(c); o = np.stack([C8[ix[c], V8.index("body_hm")], C8[ix[c], V8.index("body_tor")], C5[ix[c], V5.index("legs12")]]); print(c, "flip", f8, f5, bool(FL["flip"][IX[c]]), "max pixel difference from the saved crops", int(np.abs(x.astype(int) - o.astype(int)).max()))
    sys.exit()
OV = os.environ["Q4_OVL"]; names = [f[:-4] for f in os.listdir(f"{OV}/kpts")]; todo = []
for c in names:
    a = pickle.load(open(f"{W}/kpts/{c}.pkl", "rb")); b = pickle.load(open(f"{OV}/kpts/{c}.pkl", "rb")); ba, bb = np.asarray(a["box"])[-1], np.asarray(b["box"])[-1]; d = np.hypot((ba[0] + ba[2] - bb[0] - bb[2]) / 2, (ba[1] + ba[3] - bb[1] - bb[3]) / 2) / max(bb[3] - bb[1], 1)
    if d > 0.3: todo.append((c, float(d)))
print(f"overlay {OV}: {len(names)} clips, moved more than 0.3 h: {len(todo)} (public {sum(c not in prv for c, _ in todo)}, private {sum(c in prv for c, _ in todo)})", flush=True); out, ok_c, nflip, ex = [], [], 0, []
for c, d in todo:
    r = three(c)
    if r is None: print("  no crop", c); continue
    nflip += int(r[0] != bool(FL["flip"][IX[c]])); ok_c.append(c); out.append(r[2])
    if c not in prv and len(ex) < 10: ex.append(np.hstack([C8[ix[c], V8.index("body_hm")], r[2][0], C5[ix[c], V5.index("legs12")], r[2][2]]))
np.savez(f"{PC}/cropsfix_{TAG}.npz", clips=np.array(ok_c), crops=np.stack(out), dist=np.array([d for c, d in todo if c in ok_c])); cv2.imwrite(f"{P}/logs/cropsfix_{TAG}.jpg", cv2.cvtColor(np.vstack(ex), cv2.COLOR_RGB2BGR)); print(f"saved {len(ok_c)} clips; flip differs from the saved flip on {nflip}", flush=True)
