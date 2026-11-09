"""Late-frame reader (overnight Job 1 in OVERNIGHT_JOBS.md). 12-20% of clips end at or just after contact, with the ball (or its blur streak) leaving
the spot. Label-free and per clip: the same rules run on public and private clips.
run:  per clip, the last 3 native frames and one reference frame (the latest of the last 8 where the spot patch still looks like the ball, contact.pkl)
      are warped to last-frame coordinates (ballspot homographies) into a region around the spot scaled so the kicker is 120 px tall. New bright pixels
      = min(R,G,B) of the frame minus a 7x7 max filter of the reference (the ball is white; the max filter absorbs small misalignments of pitch lines).
      Masked: the kicker's skeleton in both frames (RTMW, thick limbs), the spot itself, pixels outside either image. Connected components are the
      candidates. Each candidate's image displacement from the spot is decomposed on the NBJW camera's image directions of the pitch axes at the spot
      (lateral l = kicker's left, forward f = toward the goal, up u), with the height tied to the forward travel (u = 0.1 f, a low-to-mid shot), giving
      a lateral and forward component and an angle (+ = kicker's left).
eval: on public clips, which candidates are the ball (eye-check montages, labels hidden), how often the reader fires, and how its direction agrees with
      kick_direction vs P4rkT out-of-fold on the same clips; private = automatic counts only.
usage: python late_reader.py run [clip,clip -> debug images only] | eval  -> cache/late_reader.pkl, logs/late_reader.txt, logs/late_*.jpg"""
import sys, time, pickle, numpy as np, cv2
from multiprocessing import Pool
from common import *

Z, HU = 120.0, 0.1                                     # ROI scale (kicker height -> px), assumed rise per metre of forward travel
X0 = pickle.load(open(f"{PC}/contact.pkl", "rb")); B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb")); KCK = pickle.load(open(f"{PC}/kickcheck.pkl", "rb"))
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
FEETL = [(15, 17), (15, 19), (17, 18), (16, 20), (16, 22), (20, 21)]


def calib(c, b):
    """last-frame 3x4 pitch -> image projection and the goal-end sign, or None"""
    fp = f"{PC}/nbjw/{c}.pkl"; k = KCK.get(c)
    if not os.path.exists(fp) or k is None: return None
    r = pickle.load(open(fp, "rb"))
    if r is None or r["rep_err"] >= 10: return None
    Pl = np.asarray(b["M"][r["frame"]], float) @ np.asarray(r["P"], float) if r["frame"] != len(b["F"]) - 1 else np.asarray(r["P"], float)
    return Pl, float(k["s"])


def axes(Pl, s):
    """image directions (px per metre) of the pitch axes at the spot: lateral (kicker's left), forward (to goal), up"""
    pr = lambda X: (lambda q: q[:2] / q[2])(Pl @ np.r_[X, 1.0]); o = np.array([41.5 * s, 0.0, 0.0]); p0 = pr(o)
    Il, If = pr(o + [0, -s, 0]) - p0, pr(o + [s, 0, 0]) - p0; Iu = pr(o + [0, 0, 1.0]) - p0
    if Iu[1] > 0: Iu = pr(o + [0, 0, -1.0]) - p0                                       # z sign convention: up must go up in the image
    return Il, If, Iu


ZT = (0.1, 1.0, 2.3)                                     # heights (m) where the ball might cross the goal line: ground, mid, crossbar. One image
                                                         # point fixes the flight only up to a lateral-vs-height trade-off (strong in side-on views),
                                                         # so a read counts only if its side is the same at all three heights


def target(Pl, s, p, zt):
    """straight flight from the spot (41.5 s, 0, 0.11) to the goal line (52.5 s, y, zt): the image point p fixes the travelled fraction u and y
    (linear in u and w = u y). Returns (u, lateral at the goal line in m, + = kicker's left). A 2D ground-plane decomposition is ill-conditioned in
    oblique views (lateral and forward map to nearly the same image direction), this is not."""
    up = Pl @ np.r_[41.5 * s, 0, 1.0, 1]; o = Pl @ np.r_[41.5 * s, 0, 0, 1]; zs = 1.0 if up[1] / up[2] < o[1] / o[2] else -1.0   # z sign: up is up in the image
    S = np.r_[41.5 * s, 0.0, 0.11 * zs, 1.0]; a, bu, bw = Pl @ S, Pl @ np.r_[11.0 * s, 0.0, (zt - 0.11) * zs, 0.0], Pl @ np.r_[0.0, 1.0, 0.0, 0.0]
    M = np.array([[bu[0] - p[0] * bu[2], bw[0] - p[0] * bw[2]], [bu[1] - p[1] * bu[2], bw[1] - p[1] * bw[2]]])
    if abs(np.linalg.det(M)) < 1e-9: return np.nan, np.nan
    u, w = np.linalg.solve(M, -np.array([a[0] - p[0] * a[2], a[1] - p[1] * a[2]]))
    return float(u), float(-s * w / u) if abs(u) > 1e-6 else np.nan


def kmask(kp, sc, T, shape):
    """the kicker's skeleton, thick, in ROI coordinates"""
    m = np.zeros(shape, np.uint8); q = warp(T, kp); ok = sc >= 1.0
    for (a, b_), t in [(l, int(0.15 * Z)) for l in LIMBS] + [(l, int(0.10 * Z)) for l in FEETL]:
        if ok[a] and ok[b_]: cv2.line(m, tuple(int(v) for v in q[a]), tuple(int(v) for v in q[b_]), 255, t)
    if ok[[5, 6, 12, 11]].all(): cv2.fillPoly(m, [q[[5, 6, 12, 11]].astype(np.int32)], 255)
    if ok[0]: cv2.circle(m, tuple(int(v) for v in q[0]), int(0.15 * Z), 255, -1)
    for j in range(17, 23):
        if ok[j]: cv2.circle(m, tuple(int(v) for v in q[j]), int(0.06 * Z), 255, -1)
    return cv2.dilate(m, np.ones((7, 7), np.uint8))


def one(c, debug=False):
    try:
        b, k, x = load("ballspot", c), load("kpts", c), X0.get(c)
        if b is None or k is None or x is None or not B3.get(c) or not np.isfinite(B3[c]["ball"][0]): return c, None
        cal = calib(c, b)
        if cal is None: return c, None
        Pl, s_ = cal; F = list(b["F"]); h = float(x["h"]); sp = np.asarray(B3[c]["ball"], float)
        last8 = [F.index(fr) for fr in x["frames"]]; q = x["ncc"]
        good = [t for t in range(len(last8) - 2) if q[t] >= 0.7]                         # ball still on the spot, at least 2 frames before the end
        tr = good[-1] if good else 0; ri = last8[tr]; use = last8[-3:]
        refs = [ri] + [j for j in range(ri - 1, -1, -1)][:3]                              # background = the reference and up to 3 earlier frames
        Il, If, Iu = axes(Pl, s_); A = np.c_[Il, If + HU * Iu]
        p0 = (lambda q_: q_[:2] / q_[2])(Pl @ np.r_[41.5 * s_, 0.0, 0.0, 1.0])
        if abs(np.linalg.det(A)) < 1e-6: return c, None
        z = Z / h; W_, H_ = int(6 * Z), int(4 * Z); T0 = np.array([[z, 0, -(sp[0] - 3 * h) * z], [0, z, -(sp[1] - 2.5 * h) * z], [0, 0, 1.0]])
        spR = (sp - (sp - [3 * h, 2.5 * h])) * z                                          # spot in ROI px
        fr = read_frames(c, [F[i] for i in refs + use]); wh = (fr[0].shape[1], fr[0].shape[0])
        def wp(f, i):
            T = T0 @ np.asarray(b["M"][i], float)
            im = cv2.warpPerspective(f, T, (W_, H_), flags=cv2.INTER_LINEAR, borderValue=0)
            v = cv2.warpPerspective(np.full((wh[1], wh[0]), 255, np.uint8), T, (W_, H_), flags=cv2.INTER_NEAREST, borderValue=0)
            return im, v, T
        # background: per channel, the max over the reference frames of a 7x7 max filter (absorbs small misalignments, and the kicker and his shadow
        # move between the references, so the grass under them shows up in at least one). New = brighter than that in any channel (white or coloured
        # ball; shadows only get darker).
        WR = [wp(f, i) for f, i in zip(fr[:len(refs)], refs)]; R, vR, TR = WR[0]
        IB = np.max([cv2.dilate(w[0], np.ones((7, 7), np.uint8)) for w in WR], 0).astype(np.int16); mR = kmask(k["kp"][ri], k["sc"][ri], TR, (H_, W_))
        spot_m = np.zeros((H_, W_), np.uint8); cv2.circle(spot_m, tuple(int(v) for v in spR), int(0.12 * Z), 255, -1)
        # ball and grass colours from the reference frame: the ball = the spot centre, the grass = a ring around it away from the kicker
        yy, xx = np.mgrid[0:H_, 0:W_]; rr = np.hypot(xx - spR[0], yy - spR[1]) / Z
        bc = R[(rr < 0.04) & (vR > 0)].reshape(-1, 3).mean(0); ring = (rr > 0.2) & (rr < 0.4) & (mR == 0) & (vR > 0)
        gc = np.median(R[ring].reshape(-1, 3), 0) if ring.sum() > 20 else np.full(3, np.nan); bg = bc - gc
        alpha = lambda col: float((col - gc) @ bg / max(bg @ bg, 1e-6))                 # 0 = grass colour, 1 = ball colour (a blurred ball mixes)
        out = dict(fps=b["fps_native"], h=h, ref_t=tr, ncc=np.asarray(q), Il=Il, If=If, Iu=Iu, s=s_, frames=[], ballcol=bc, grass=gc)
        dbg = []
        for t, (i, f) in enumerate(zip(use, fr[len(refs):])):
            im, v, T = wp(f, i)
            D = (im.astype(np.int16) - IB).max(2); I = im.max(2).astype(np.int16)
            bad = (vR == 0) | (v == 0) | (mR > 0) | (kmask(k["kp"][i], k["sc"][i], T, (H_, W_)) > 0) | (spot_m > 0)
            bw = ((D > 30) & ~bad).astype(np.uint8); bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8); cands = []
            spot_a = alpha(im[(rr < 0.04) & (v > 0)].reshape(-1, 3).mean(0))               # is the spot still the ball's colour?
            kq = warp(T, k["kp"][i][[11, 12, 13, 14]]); ks = k["sc"][i][[11, 12, 13, 14]]
            hip_y = float(np.mean(kq[:2, 1][ks[:2] >= 1.0])) if (ks[:2] >= 1.0).any() else np.nan
            knee_y = float(np.mean(kq[2:, 1][ks[2:] >= 1.0])) if (ks[2:] >= 1.0).any() else np.nan
            for j in range(1, n):
                ar = st[j, 4]
                if ar < 12 or ar > 2500: continue
                ys, xs = np.nonzero(lab == j); d = (cen[j] - spR) / Z                    # displacement in kicker heights (image)
                dist = float(np.hypot(*d))
                if dist < 0.15 or dist > 3.0: continue
                cov = np.cov(np.c_[xs, ys].T) if ar > 2 else np.eye(2); ev, evec = np.linalg.eigh(cov); el = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))
                ax_ = evec[:, 1]; radial = float(abs(ax_ @ (d / max(dist, 1e-6))))           # streak axis vs the ray from the spot (1 = along it)
                lat, fwd = np.linalg.solve(A, d * h)                                       # old 2D decomposition, kept for comparison
                tg = [target(Pl, s_, p0 + d * h, zt) for zt in ZT]                         # p0 = projected spot: displacement from the ball, applied there
                u1, l1 = tg[1]
                if not os.environ.get("LR_ALL") and (not (0.0 < u1 < 0.8) or not np.isfinite(l1)): continue   # toward the goal (players behind: u < 0)
                con = float(D[ys, xs].mean())
                cands.append(dict(dx=float(d[0]), dy=float(d[1]), dist=dist, area=int(ar), con=con, el=el, radial=radial, lat2d=float(lat), fwd2d=float(fwd),
                                  u=u1, lat=l1, lat_lo=tg[0][1], lat_hi=tg[2][1], ang=float(np.degrees(np.arctan2(l1, 11.0))), bright=float(I[ys, xs].mean()),
                                  ca=alpha(im[ys, xs].reshape(-1, 3).mean(0)), below_hip=float(cen[j][1] - hip_y) / Z if np.isfinite(hip_y) else np.nan,
                                  below_knee=float(cen[j][1] - knee_y) / Z if np.isfinite(knee_y) else np.nan))
            cands.sort(key=lambda e: -e["con"] * np.sqrt(min(e["area"], 400)))
            out["frames"].append(dict(t=len(last8) - 3 + t, cands=cands[:8], n=len(cands), spot_a=spot_a))
            if debug: dbg.append((im, bw, cands[:3]))
        if debug:
            rows = []
            for im, bw, cs in dbg:
                a = im.copy(); a[bw > 0] = (0, 0, 255); cv2.circle(a, tuple(int(v) for v in spR), int(0.12 * Z), (0, 255, 0), 1)
                for e in cs:
                    cv2.circle(a, (int(spR[0] + e["dx"] * Z), int(spR[1] + e["dy"] * Z)), 10, (255, 0, 255), 2)
                    cv2.putText(a, f"{e['ang']:.0f} c{e['con']:.0f} a{e['area']}", (int(spR[0] + e["dx"] * Z) + 12, int(spR[1] + e["dy"] * Z)), 0, 0.45, (255, 255, 255), 1)
                rows.append(np.hstack([im, a]))
            cv2.imwrite(f"{P}/logs/late_dbg_{c[:4]}.jpg", np.vstack(rows + [np.hstack([R, cv2.cvtColor(mR, cv2.COLOR_GRAY2BGR)])]))
        return c, out
    except Exception as e:
        return c, dict(err=repr(e))


if __name__ == "__main__":
    if sys.argv[1] == "run" and len(sys.argv) > 2:
        for c_ in sys.argv[2].split(","):
            c_ = [cc for cc in CLIPS if cc.startswith(c_)][0]; _, r = one(c_, debug=True)
            if r is None or "err" in r: print(c_, r); continue
            print(c_, "fps", round(r["fps"], 1), "ref_t", r["ref_t"], "ncc", np.round(r["ncc"], 2))
            for fr_ in r["frames"]: print("   t", fr_["t"], "n", fr_["n"], [dict(u=round(e["u"], 2), lat=(round(e["lat_lo"], 1), round(e["lat"], 1), round(e["lat_hi"], 1)), con=round(e["con"]), a=e["area"], d=(round(e["dx"], 2), round(e["dy"], 2))) for e in fr_["cands"][:3]])
    elif sys.argv[1] == "run":
        t0 = time.time()
        with Pool(int(os.environ.get("NPOOL", "12"))) as p: R_ = dict(p.map(one, CLIPS, chunksize=8))
        pickle.dump(R_, open(f"{PC}/late_reader.pkl", "wb"))
        err = [c for c, r in R_.items() if r is not None and "err" in r]
        print(f"done {len(R_)} clips in {time.time() - t0:.0f}s | none {sum(r is None for r in R_.values())} | errors {len(err)}", err[:5])
