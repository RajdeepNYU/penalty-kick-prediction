"""Stage R: ground-plane (rectified) kicker features from the per-clip camera (camera2.pkl) and the new ball.
Only quantities a single elevated view can measure are rectified: ground-contact points (feet), same-height differences (hip / shoulder / ear axes),
and displacements of roughly constant-height points over time (hip / shoulder / head trajectories). Segment leans are kept as image-x leans and
re-weighted by the camera azimuth (a lean along the camera's X axis = cos(phi) * forward - sin(phi) * lateral).
Frame: u = towards goal, v = kicker's left (mirror negates v, flips the foot sign and the azimuth). Units: metres.
build(c, stream, mirror) -> (static vector, per-frame channels [T, C], static-for-NN vector). usage: [CAMF=camera2_old] python kinR.py STREAM  (A, B, E, F, G as in streams.py)"""
import os, sys, pickle, warnings, numpy as np
from common import *
warnings.filterwarnings("ignore")
from kin2 import legs_at, match2, slope

H_M, T, CT = 1.78, 12, 2.0
Z = dict(ank=0.08, toe=0.03, heel=0.05, hip=0.95, sho=1.45, nose=1.65)
NAMES = []


def legs(P, sc, nF):
    L = [legs_at(P[j], sc[j]) for j in range(nF)]; order = [[0, 1]]
    cen = lambda leg: np.nanmean(np.array([leg[q] for q in ("ank", "toe", "heel")]), axis=0) if np.isfinite(leg["ank"]).all() else np.array([np.nan, np.nan])
    for t in range(1, nF):
        ref = [cen(L[t - 1][order[-1][i]]) for i in range(2)]; cand = np.array([cen(L[t][j]) for j in range(2)]); m = match2(cand, None, np.array(ref))
        o = [m[0] if m[0] is not None else 0, m[1] if m[1] is not None else 1]; order.append(o if o[0] != o[1] else [0, 1])
    ident = [[L[t][order[t][i]] for t in range(nF)] for i in range(2)]; n6 = min(6, nF)
    def travel(i):
        pts = np.array([ident[i][t]["ank"] for t in range(nF - n6, nF)]); d = np.hypot(*np.diff(pts, axis=0).T); return np.nansum(d) if np.isfinite(d).any() else 0.0
    tv = [travel(0), travel(1)]; ki = int(np.argmax(tv))
    return {"kk": ident[ki], "pl": ident[1 - ki]}, tv[ki], tv[1 - ki]


def build(c, stream, mirror, cam, b, kb, box):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(np.asarray(vals, np.float32)).ravel(); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; kp, sc = kb["kp"], kb["sc"]; M = b["M"]
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    Pm = np.where((sc >= CT)[..., None], P, np.nan)
    hL = max(float(box[-1][3] - box[-1][1]), 20.0); Ai = cam["Ai"]; bg = cam["ball_ground"]; az = cam["az"]; sg = -1.0 if mirror else 1.0
    phi = np.radians(cam["phi"]); cph, sph = np.cos(phi) * sg, np.sin(phi)
    G = lambda X, z: ((np.asarray(X, float) - bg - az * z) @ Ai.T) * np.array([1.0, sg])          # point at height z -> ground (u, v) m
    Gd = lambda V: (np.asarray(V, float) @ Ai.T) * np.array([1.0, sg])                           # displacement between equal-height points
    lg, tk, tp = legs(P, sc, nF)
    arr = lambda w, nm: np.array([lg[w][t][nm] for t in range(nF)])
    feet = {(w, nm): G(arr(w, nm), Z[nm]) for w in ("kk", "pl") for nm in ("ank", "toe", "heel")}
    hm, sm, nose = np.nanmean(Pm[:, [11, 12]], 1), np.nanmean(Pm[:, [5, 6]], 1), Pm[:, 0]
    l3 = slice(max(0, nF - 3), nF); l4 = list(range(max(0, nF - 4), nF)); l6 = list(range(max(0, nF - 6), nF))
    near = lambda o: int(np.abs(ts + o).argmin())
    # ---- camera
    add("cam", [cam["theta"], cph, sph, float(cam["posts"] and not cam.get("posts_rejected", 0)), cam["av_err"] if np.isfinite(cam["av_err"]) else -1])
    # ---- plant foot (ground contact)
    for nm in ("ank", "toe", "heel"): add(f"pl_{nm}_g", feet[("pl", nm)][-1])
    pc = np.nanmean(np.stack([feet[("pl", nm)][l3] for nm in ("ank", "toe", "heel")]), axis=(0, 1)); add("pl_cen_g", pc); add("pl_cen_dist", np.hypot(*pc))
    th_ = feet[("pl", "toe")] - feet[("pl", "heel")]; ang = np.arctan2(th_[:, 1], th_[:, 0])
    add("pl_dir_last", [np.cos(ang[-1]), np.sin(ang[-1])]); add("pl_dir_m3", [np.nanmean(np.cos(ang[l3])), np.nanmean(np.sin(ang[l3]))])
    # ---- kicking foot
    for nm in ("ank", "toe"): add(f"kk_{nm}_g", feet[("kk", nm)][-1])
    kc = np.nanmean(np.stack([feet[("kk", nm)][-1] for nm in ("ank", "toe", "heel")]), 0); add("kk_minus_pl", kc - pc)
    for nm in ("ank", "toe"):
        y = feet[("kk", nm)]; vu, vv = slope(ts[l4], y[l4, 0]), slope(ts[l4], y[l4, 1]); add(f"kk_{nm}_vel", [vu, vv, np.hypot(vu, vv), np.arctan2(vv, vu) if np.isfinite(vu) else np.nan])
    add("travel", [tk / hL, tp / hL])
    # ---- body axes in the ground plane (angle mod 180)
    for nm, (a_, b_) in (("hipax", (11, 12)), ("shax", (5, 6)), ("earax", (3, 4))):
        V = Gd(Pm[:, b_] - Pm[:, a_]); a2 = 2 * np.arctan2(V[:, 1], V[:, 0]); n = np.hypot(V[:, 0], V[:, 1])
        full = np.full(nF, np.nan); fin = np.isfinite(a2)
        if fin.sum() > 1: full[fin] = np.unwrap(a2[fin])
        add(f"{nm}_g_last", [np.cos(a2[-1]), np.sin(a2[-1]), n[-1]]); add(f"{nm}_g_m3", [np.nanmean(np.cos(a2[l3])), np.nanmean(np.sin(a2[l3]))]); add(f"{nm}_g_angvel", slope(ts[l6], full[l6]))
    # ---- trajectories (displacement relative to the last frame) and body position relative to ball / plant foot
    for nm, S, z in (("hip", hm, Z["hip"]), ("sho", sm, Z["sho"]), ("nose", nose, Z["nose"])):
        D = Gd(S - S[-1]); add(f"{nm}_disp", np.concatenate([D[near(o)] for o in (0.15, 0.25, 0.45, 0.8)]))
        Gp = G(S, z); add(f"{nm}_pos_g", Gp[-1]); add(f"{nm}_over_plant", Gp[-1] - pc)
        for w in (0.3, 0.8):
            m = np.where(ts >= -w)[0]; add(f"{nm}_vel{w}", [slope(ts[m], D[m, 0]), slope(ts[m], D[m, 1])])
    D = Gd(hm - hm[-1])
    for a, b_ in ((0.8, 0.3), (0.3, 0.0)):
        v = D[near(b_)] - D[near(a)]; add(f"approach_{a}", [np.arctan2(v[1], v[0]), np.hypot(*v)])
    # ---- segment leans along the camera X axis, re-weighted by azimuth
    segs = {"trunk": (hm, sm), "head": (sm, nose), "plsh": (arr("pl", "knee"), arr("pl", "ank")), "plth": (arr("pl", "hip"), arr("pl", "knee")),
            "kksh": (arr("kk", "knee"), arr("kk", "ank")), "kkth": (arr("kk", "hip"), arr("kk", "knee"))}
    for nm, (a_, b_) in segs.items():
        d = b_ - a_; lx = sg * d[:, 0] / np.maximum(np.abs(d[:, 1]), 0.05 * hL)
        if nm in ("plsh", "plth", "kksh", "kkth"): lx = -lx                                       # leg segments point downwards: lean = top relative to bottom
        add(f"lean_{nm}", [lx[-1], np.nanmean(lx[l3]), -sph * lx[-1], -sph * np.nanmean(lx[l3]), cph * lx[-1], cph * np.nanmean(lx[l3])])
    meta = META[c]; add("foot_sign", (1.0 if meta["foot"] == "right" else -1.0) * sg)
    st = np.array(f_, np.float32)
    # ---- per-frame channels for the sequence network
    ch = [feet[(w, nm)] for w in ("kk", "pl") for nm in ("ank", "toe", "heel")]
    ch += [Gd(S - S[-1]) for S in (hm, sm, nose)] + [G(hm, Z["hip"])]
    for a_, b_ in ((11, 12), (5, 6)):
        V = Gd(Pm[:, b_] - Pm[:, a_]); a2 = 2 * np.arctan2(V[:, 1], V[:, 0]); ch.append(np.stack([np.cos(a2), np.sin(a2)], -1))
    for nm, (a_, b_) in segs.items():
        d = b_ - a_; lx = sg * d[:, 0] / np.maximum(np.abs(d[:, 1]), 0.05 * hL); lx = -lx if nm in ("plsh", "plth", "kksh", "kkth") else lx; ch.append(np.stack([-sph * lx, cph * lx], -1))
    X = np.concatenate(ch, 1).astype(np.float32); fin = np.isfinite(X).mean(1, keepdims=True).astype(np.float32); X = np.clip(np.nan_to_num(X, nan=0.0), -20, 20)
    X = np.concatenate([X, ts[:, None].astype(np.float32), fin, np.ones((nF, 1), np.float32)], 1); seq = np.zeros((T, X.shape[1]), np.float32); n = min(T, nF); seq[-n:] = X[-n:]
    nst = np.array([(1.0 if meta["foot"] == "right" else -1.0) * sg, 1.0 if meta["gender"] == "M" else 0.0, meta["fps"] / 60.0, cam["theta"] / 20.0, cph, sph], np.float32)
    return st, seq, nst


stream = sys.argv[1] if len(sys.argv) > 1 else "A"; camf = os.environ.get("CAMF", "camera2"); CAM = pickle.load(open(f"{PC}/{camf}.pkl", "rb"))


def one(c):
    from streams import stream_kb
    b = load("ballspot", c); kb = stream_kb(c, stream); cam = CAM.get(c)
    if b is None or kb is None or cam is None: return c, None, None
    NAMES.clear(); a = build(c, stream, False, cam, b, kb, kb["box"]); names = list(NAMES); NAMES.clear(); m = build(c, stream, True, cam, b, kb, kb["box"])
    return c, (a, m), names


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, CLIPS, chunksize=20)
    rows = {c: r for c, r, _ in res if r is not None}; names = next(n for _, r, n in res if r is not None)
    pickle.dump(dict(rows=rows, names=names), open(f"{PC}/kinR_{stream}{'' if camf == 'camera2' else '_' + camf}.pkl", "wb"))
    print(f"stream {stream}: built {len(rows)} skipped {len(CLIPS) - len(rows)} | static dim {len(names)} | seq {rows[next(iter(rows))][0][1].shape}")
