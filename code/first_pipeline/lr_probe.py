import pickle, os, numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(D + r"\train_ground_truth.csv"); sc_all = []; rows = []
for c, foot in zip(gt["clip"][::4], gt["preferred_foot"][::4]):
    bp, kp_ = f"{C}/ballspot/{c}.pkl", f"{C}/kpts/{c}.pkl"
    if not (os.path.exists(bp) and os.path.exists(kp_)): continue
    b = pickle.load(open(bp, "rb")); k = pickle.load(open(kp_, "rb"))
    if b is None or k is None: continue
    M = b["M"]; kp = k["kp"]; sc = k["sc"]; sc_all.append(sc[-3:, [15, 16, 17, 19, 20, 22]].ravel())
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    h = k["box"][-1][3] - k["box"][-1][1]; A = P[-6:, [15, 16]]          # [6,2 feet,2]
    sw = 0
    for t in range(5):
        a, bb = A[t], A[t + 1]
        if np.hypot(*(a[0] - bb[1])) + np.hypot(*(a[1] - bb[0])) < 0.7 * (np.hypot(*(a[0] - bb[0])) + np.hypot(*(a[1] - bb[1]))): sw += 1
    # toe/heel coherence: toe of the labelled foot nearer to its own ankle than to the other ankle?
    coh = np.mean([np.hypot(*(P[-1, 17] - P[-1, 15])) < np.hypot(*(P[-1, 17] - P[-1, 16])), np.hypot(*(P[-1, 20] - P[-1, 16])) < np.hypot(*(P[-1, 20] - P[-1, 15]))])
    # travel with raw labels vs with temporally de-swapped identities
    raw = {n: np.hypot(*np.diff(A[:, i], axis=0).T).sum() / h for i, n in enumerate("LR")}
    ids = [A[0].copy()]
    for t in range(1, 6):
        prev, cur = ids[-1], A[t]
        keep = np.hypot(*(prev[0] - cur[0])) + np.hypot(*(prev[1] - cur[1])); swp = np.hypot(*(prev[0] - cur[1])) + np.hypot(*(prev[1] - cur[0]))
        ids.append(cur[::-1].copy() if swp < keep else cur.copy())
    ids = np.array(ids); trv = {n: np.hypot(*np.diff(ids[:, i], axis=0).T).sum() / h for i, n in enumerate("LR")}
    pref = "R" if foot == "right" else "L"
    rows.append(dict(swaps=sw, coh=coh, agree_raw=("R" if raw["R"] > raw["L"] else "L") == pref, agree_deswap=("R" if trv["R"] > trv["L"] else "L") == pref, ratio=max(trv.values()) / max(min(trv.values()), 1e-6)))
r = pd.DataFrame(rows); s = np.concatenate(sc_all)
print("RTMW score distribution (ankle/toe/heel, last 3 frames) percentiles 1/5/25/50/75/95/99:", np.percentile(s, [1, 5, 25, 50, 75, 95, 99]).round(2))
print("clips", len(r), "| mean label swaps per clip (of 5 steps): %.2f ; clips with any swap: %.2f" % (r.swaps.mean(), (r.swaps > 0).mean()))
print("toe/heel belongs to labelled ankle: %.3f" % r.coh.mean())
print("kicking foot (more travel) == preferred_foot: raw labels %.3f | after de-swapping identities %.3f" % (r.agree_raw.mean(), r.agree_deswap.mean()))
for q in (1.5, 2.0, 3.0): m = r.ratio > q; print(f"   when one foot travels >{q}x the other (n={m.sum()}): agreement {r.agree_deswap[m].mean():.3f}")
