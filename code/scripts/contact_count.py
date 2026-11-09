"""Does the clip end after contact? (TA tips notebook: 0058_public's last two frames are at / after contact.) Counts the clips whose last frames show
the ball leaving the spot, which would make the ball's own direction a kick cue. Read-only on the Q4_work caches.
run:  YOLO11x sports-ball detections (conf >= 0.01) on spot-centred crops (side 3 kicker heights, upscaled to 640) of the last 8 native frames,
      warped to last-frame coordinates with the ballspot camera homographies; spot = ball3 (ball2, or the NBJW spot on the 79 moved clips);
      nearest foot keypoint (RTMW, slots 15-22) to the spot per frame -> cache/contact.pkl
eval: per clip, in units of kicker height h: ON = a detection within 0.07 h of the spot (ball diameter ~0.12 h); OFF = a detection 0.10-2.0 h away that
      no frame >= 3 earlier has within 0.05 h (not a static ball-like object). Classes from the last frame: on_spot / moved (ball seen on the spot
      earlier, not on it now, a new detection off it) / gone (seen earlier, nowhere now: foot occlusion, blur or out of the crop) / unknown
      (never seen on the spot). Counts public vs private and by fps, P4rkT out-of-fold kick accuracy per class (the ceiling), montages of
      the public moved / gone clips (logs/contact_*.jpg) for an eye check.
usage: python contact_count.py run | eval   -> logs/contact_count.txt"""
import sys, time, pickle, numpy as np
from common import *

R_ON, R_OFF0, R_OFF1, R_STATIC = 0.07, 0.10, 2.0, 0.05
stage = sys.argv[1]

if stage == "run":
    from ultralytics import YOLO
    yolo = YOLO(ROOT + r"\Q1-Q3work\yolo11x.pt")
    B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb")); out, t0 = {}, time.time()
    todo = CLIPS if len(sys.argv) < 3 else sys.argv[2].split(",")             # optional comma-separated clip list (test run, not saved)
    PART = f"{PC}/contact_part.pkl"                                          # checkpoint every 200 clips; a rerun resumes from it
    if todo is CLIPS and os.path.exists(PART): out = pickle.load(open(PART, "rb")); print("resuming from", len(out), "clips", flush=True)
    for n, c in enumerate(todo):
        if c in out: continue
        b, k = load("ballspot", c), load("kpts", c)
        if b is None or not B3.get(c) or not np.isfinite(B3[c]["ball"][0]): out[c] = None; continue
        F = list(b["F"]); last = F[-1]; idx = [i for i, j in enumerate(F) if j > last - 8]; h = float(b["boxes"][-1][3] - b["boxes"][-1][1])
        s = np.asarray(B3[c]["ball"], float); fr = read_frames(c, [F[0]] + [F[i] for i in idx]); f0, fr = fr[0], fr[1:]; crops, meta = [], []
        # appearance at the spot: 24x24 grey patch covering +-0.12 h around the spot in last-frame coords, sampled in each frame through its homography
        g = (np.mgrid[0:24, 0:24][::-1].reshape(2, -1).T - 11.5) * (0.24 * h / 24) + s
        def patch(f, i):
            m = warp(np.linalg.inv(b["M"][i]), g).astype(np.float32)
            return cv2.remap(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32), m[:, 0].reshape(24, 24), m[:, 1].reshape(24, 24), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        pref = patch(f0, 0); pats = [patch(f, i) for i, f in zip(idx, fr)]
        for i, f in zip(idx, fr):
            sx, sy = warp(np.linalg.inv(b["M"][i]), s); side = 3.0 * h; x0, y0 = int(sx - side / 2), int(sy - side / 2)
            pad = cv2.copyMakeBorder(f, 600, 600, 600, 600, cv2.BORDER_REPLICATE)
            crops.append(cv2.resize(pad[y0 + 600:y0 + 600 + int(side), x0 + 600:x0 + 600 + int(side)], (640, 640), interpolation=cv2.INTER_CUBIC)); meta.append((x0, y0, side))
        res = yolo.predict(crops, imgsz=640, conf=0.01, classes=[32], verbose=False, device=0, batch=len(crops))
        dets, foot = [], []
        for i, r, (x0, y0, side) in zip(idx, res, meta):
            bb = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); sc = side / 640
            xy = np.c_[(bb[:, 0] + bb[:, 2]) / 2 * sc + x0, (bb[:, 1] + bb[:, 3]) / 2 * sc + y0] if len(bb) else np.zeros((0, 2))
            xy = warp(b["M"][i], xy) if len(xy) else xy
            dets.append(np.c_[(xy - s) / h, cf, (bb[:, 2] - bb[:, 0]) * sc / h] if len(bb) else np.zeros((0, 4)))
            if k is not None and i < len(k["kp"]):
                kp = warp(b["M"][i], k["kp"][i, 15:23]); ok = k["sc"][i, 15:23] > 0.3
                foot.append(float(np.min(np.hypot(*((kp[ok] - s) / h).T))) if ok.any() else np.nan)
            else: foot.append(np.nan)
        ncc = lambda a, q: float(((a - a.mean()) * (q - q.mean())).sum() / (np.sqrt(((a - a.mean()) ** 2).sum() * ((q - q.mean()) ** 2).sum()) + 1e-6))
        out[c] = dict(frames=[F[i] for i in idx], dets=dets, foot=np.array(foot), h=h, fps=b["fps_native"], ncc=np.array([ncc(pref, q) for q in pats]),
                      foot0=float(np.nanmin(foot[:1])) if foot else np.nan)
        if (n + 1) % 200 == 0:
            print(n + 1, f"{time.time() - t0:.0f}s elapsed", flush=True)
            if todo is CLIPS: pickle.dump(out, open(PART, "wb"))
    if todo is CLIPS: pickle.dump(out, open(f"{PC}/contact.pkl", "wb")); print("done", len(out))
    else:
        for c, r in out.items():
            print(c, "h", round(r["h"]), "fps", round(r["fps"], 1), "foot", np.round(r["foot"], 2), "ncc", np.round(r["ncc"], 2))
            for f, d in zip(r["frames"], r["dets"]): print("   frame", f, np.round(d[np.argsort(-d[:, 2])][:3], 3).tolist())

else:
    from cvlib import cl, yk
    from blend_eval import trP, bagP
    fx = f"{PC}/contact.pkl" if os.path.exists(f"{PC}/contact.pkl") else f"{PC}/contact_part.pkl"     # partial checkpoint while a run is going
    yk = np.asarray(yk, dtype=object); X = pickle.load(open(fx, "rb")); print(f"{os.path.basename(fx)}: {len(X)} clips")

    def classify(r):
        """appearance first: early = best similarity of the spot patch (frames L-7..L-5) to the one ~1 s earlier; unreliable if < 0.7 (ball not
        clearly visible / aligned). changed = the last frame no longer looks like it (ncc < 0.5 and 0.3 below early); changed2 = the frame before too.
        moved = YOLO also finds a new ball off the spot in the last frame (direction available)."""
        if r is None: return "unreliable", None
        q, d, L = r["ncc"], r["dets"], len(r["ncc"]) - 1; early = q[:3].max()
        if early < 0.7: return "unreliable", None
        chg = lambda t: q[t] < 0.5 and q[t] < early - 0.3
        if not chg(L): return "stable", None
        mv = new_off(d, L)
        if mv is not None: return "moved", mv
        return ("changed2" if chg(L - 1) else "changed"), None

    def new_off(d, t):
        """the most confident YOLO ball 0.10-2.0 h off the spot in frame t that no frame >= 3 earlier has within 0.05 h (not a static object)"""
        best = None
        for e in d[t]:
            dist = np.hypot(e[0], e[1])
            if not (R_OFF0 <= dist <= R_OFF1) or e[2] < 0.15: continue
            if any(np.hypot(e[0] - q[0], e[1] - q[1]) < R_STATIC for u in range(0, max(t - 2, 0)) for q in d[u]): continue
            if best is None or e[2] > best[2]: best = e
        return best

    C = {c: classify(X.get(c)) for c in CLIPS}; pub = list(gt["clip"]); prv = list(pm["clip"])
    cats = ["stable", "changed", "changed2", "moved", "unreliable"]
    print("last-frame ball state (counts, share):")
    for nm, cs in (("public", pub), ("private", prv)):
        v = [C[c][0] for c in cs]; print(f"  {nm:8s} " + " | ".join(f"{k} {v.count(k)} ({v.count(k) / len(v):.1%})" for k in cats))
    print("by fps (public + private):")
    fps = np.array([META[c]["fps"] for c in CLIPS]); grp = np.where(fps < 20, "<20", np.where(fps < 27, "24-26", np.where(fps < 40, "30", "50-60")))
    for g in ("<20", "24-26", "30", "50-60"):
        cs = [c for c, gg in zip(CLIPS, grp) if gg == g]; v = [C[c][0] for c in cs]
        print(f"  {g:6s} n {len(cs):4d} | " + " | ".join(f"{k} {v.count(k) / max(len(v), 1):.1%}" for k in cats))
    # foot distance to the spot in the last frame, for the gone class (occlusion check)
    for k_ in ("changed", "changed2", "moved", "stable"):
        fd = np.array([X[c]["foot"][-1] for c in pub if C[c][0] == k_ and X.get(c) is not None])
        if len(fd): print(f"  public {k_:8s}: nearest foot point to the spot in the last frame, median {np.nanmedian(fd):.2f} h, share < 0.10 h {np.nanmean(fd < 0.10):.2f}")
    # P4rkT out-of-fold kick accuracy per class (same construction as kick_metrics.py)
    KC = P + r"\cache_kf"
    KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb"))); mk = np.array([c in KF for c in pub])
    B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb")); mb = np.array([bool(B3.get(c)) and B3[c].get("method") == "nbjw_spot" for c in pub])
    T2 = sum(trP(f"{S}_b2R") for S in "AFG") / 3; N2 = sum(bagP(st, q) for st in ("EF", "G") for q in ("b2", "b2RM")) / 4
    Tb = sum(trP(f"{S}_b3R") for S in "AFG") / 3; Nb = sum(bagP(st, q) for st in ("EF", "G") for q in ("b2_b3", "b2RM_b3")) / 4
    Tk = sum((np.load(f"{KC}/oof_hgb_{S}_b2R.npy") + np.load(f"{KC}/oof_et_{S}_b2R.npy")) / 2 for S in "AFG") / 3
    Pk = np.where(mk[:, None], Tk, np.where(mb[:, None], 0.5 * Tb + 0.5 * Nb, 0.5 * T2 + 0.5 * N2)); pk = cl[Pk.argmax(1)]; ok = pk == yk
    cp = np.array([C[c][0] for c in pub])
    print("P4rkT out-of-fold kick accuracy by class (public):")
    for k_ in cats:
        m = cp == k_
        if m.any(): print(f"  {k_:8s} n {m.sum():4d} acc {ok[m].mean():.3f} | wrong {int((~ok[m]).sum())} | middle share {np.mean(yk[m] == 'middle'):.2f}")
    # montages of the public flagged clips for the eye check: last 4 native frames around the spot, spot = green circle, new off-spot
    # YOLO ball = red, spot-patch similarity to ~1 s earlier printed on each tile
    for k_ in ("moved", "changed2", "changed"):
        cs = [c for c in pub if C[c][0] == k_]; rows = []
        for c in cs[:60]:
            b = load("ballspot", c); F = list(b["F"]); h = X[c]["h"]; s = np.asarray(B3[c]["ball"], float); idx = [i for i, j in enumerate(F) if j > F[-1] - 4]
            fr = read_frames(c, [F[i] for i in idx]); tiles = []
            for i, f in zip(idx, fr):
                sx, sy = warp(np.linalg.inv(b["M"][i]), s); side = 2.4 * h; x0, y0 = int(sx - side / 2), int(sy - side * 0.6)
                pad = cv2.copyMakeBorder(f, 600, 600, 600, 600, cv2.BORDER_REPLICATE); im = pad[y0 + 600:y0 + 600 + int(side), x0 + 600:x0 + 600 + int(side)].copy()
                z = 240 / im.shape[0]; im = cv2.resize(im, (240, 240), interpolation=cv2.INTER_CUBIC)
                cv2.circle(im, (int((sx - x0) * z), int((sy - y0) * z)), max(int(0.08 * h * z), 4), (0, 255, 0), 1)
                if i == idx[-1] and C[c][1] is not None:
                    e = C[c][1]; q = warp(np.linalg.inv(b["M"][i]), s + e[:2] * h); cv2.circle(im, (int((q[0] - x0) * z), int((q[1] - y0) * z)), max(int(0.08 * h * z), 4), (0, 0, 255), 2)
                cv2.putText(im, f"{X[c]['ncc'][idx.index(i) + len(X[c]['ncc']) - len(idx)]:.2f}", (6, 232), 0, 0.5, (255, 255, 255), 1); tiles.append(im)
            while len(tiles) < 4: tiles.insert(0, np.zeros((240, 240, 3), np.uint8))
            row = np.hstack(tiles); j = pub.index(c)
            cv2.putText(row, f"{c[:4]} true {yk[j]} pred {pk[j]}", (6, 20), 0, 0.6, (255, 255, 255), 2); rows.append(row)
        for p in range(0, len(rows), 12):
            cv2.imwrite(f"{P}/logs/contact_{k_}_{p // 12 + 1}.jpg", np.vstack(rows[p:p + 12]))
        print(f"wrote {(len(rows) + 11) // 12} montage(s) for public {k_} ({len(cs)} clips, first 60 shown)")
    pickle.dump({c: C[c][0] for c in CLIPS}, open(f"{PC}/contact_class.pkl", "wb"))
