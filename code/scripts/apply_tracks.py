"""Swap in tracks_new.pkl as tracks.pkl (backup: tracks_v6.pkl) and delete the downstream caches of every clip whose kicker/keeper boxes changed in ANY frame."""
import os  # notebook: paths from the environment
import pickle, os, shutil, numpy as np
ROOT = os.environ.get("Q4_ROOT", r"C:\path\to\project-root"); C = os.environ["Q4_CACHE"]
old = pickle.load(open(f"{C}/tracks.pkl", "rb")); new = pickle.load(open(f"{C}/tracks_new.pkl", "rb"))
def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)
diff = []
for c in new:
    o, n = old[c], new[c]
    if o is None or n is None or o.get("keep") is None or n.get("keep") is None: diff.append(c); continue
    for key in ("kick", "keep"):
        a, b = o[key], n[key]; fa, fb = np.isfinite(a[:, 0]), np.isfinite(b[:, 0])
        if (fa != fb).any() or any(iou(a[j], b[j]) < 0.8 for j in range(len(a)) if fa[j] and fb[j]): diff.append(c); break
print("clips whose tracks differ in any frame:", len(diff), f"({len(diff) / len(new):.1%})")
if not os.path.exists(f"{C}/tracks_v6.pkl"): shutil.copy(f"{C}/tracks.pkl", f"{C}/tracks_v6.pkl")
pickle.dump(new, open(f"{C}/tracks.pkl", "wb")); pickle.dump(diff, open(f"{C}/retrack.pkl", "wb")); n = 0
for d in ("ballspot", "kpts", "kpts3d", "kkeep", "align", "kptsC", "kptsD", "kkeep3d", "pose", "ball"):
    for c in diff:
        p = f"{C}/{d}/{c}.pkl"
        if os.path.exists(p): os.remove(p); n += 1
print("removed cached downstream files:", n)
