"""Items 1b + 7: kicker and keeper ground features from the NBJW perspective camera (cache/nbjw, nbjw_cam.py) instead of the orthographic post fit (camera2).
A point X (last-frame image coords) at height z m -> pitch (x, y) through H_z = [P0, P1, P3 - z P2]; goal frame: u = towards the goal the ball is nearer to,
origin at the ball's ground point, v = kicker's left (mirror negates v). Calibrations from the 3rd-last F frame are carried to the last frame with that
frame's M. Clips whose calibration puts the ball more than 2 m from a penalty spot get NaN features (and a quality flag), the trees fall back on kinR.
Kicker (stream S, kinR's leg identity): camera height / distance / azimuth, plant and kicking foot positions and directions, hip / shoulder / ear axes,
body trajectories and approach, as kinR but perspective-correct. Keeper (RTMW kkeep, ViTPose-L kkeepF): feet / hips / shoulders lateral position (m from the
goal centre), displacements from -0.45 s and lateral velocities.
usage: python kinN.py STREAM  -> cache/kinN_{STREAM}.pkl ; python kinN.py keep  -> cache/keepN_A.pkl, cache/keepN_F.pkl  (rows {clip: (static, mirror)}, names)"""
import sys, pickle, warnings, numpy as np
from common import *
warnings.filterwarnings("ignore")
from kin2 import slope

B2 = pickle.load(open(f"{PC}/ball2.pkl", "rb")); CT = 2.0; Z = dict(ank=0.08, toe=0.03, heel=0.05, hip=0.95, sho=1.45, ear=1.6, nose=1.65); NAMES = []
VSIGN = float(os.environ.get("NBJW_VSIGN", "-1"))       # v = VSIGN * s * y (pitch y points to the right of someone facing +x; checked against camera2 in check_kinN.py)


def camera(c, b):
    fp = f"{PC}/nbjw/{c}.pkl"
    if not os.path.exists(fp) or B2.get(c) is None: return None
    r = pickle.load(open(fp, "rb"))
    if r is None: return None
    Pl = np.asarray(b["M"][r["frame"]], float) @ r["P"] if r["frame"] != len(b["F"]) - 1 else r["P"]

    def raw(X, z):                                                                          # image point(s) at height z m -> pitch (x, y)
        X = np.asarray(X, float); Hi = np.linalg.inv(np.c_[Pl[:, 0], Pl[:, 1], Pl[:, 3] - z * Pl[:, 2]])
        g = np.c_[X.reshape(-1, 2), np.ones(X.size // 2)] @ Hi.T; return (g[:, :2] / g[:, 2:3]).reshape(X.shape)
    bg = raw(np.asarray(B2[c]["ball"], float), 0.11); s = 1.0 if bg[0] >= 0 else -1.0; spot = np.array([41.5 * s, 0.0]); dspot = float(np.hypot(*(bg - spot)))
    pos = np.array(r["cam"]["position_meters"], float)
    return dict(raw=raw, bg=bg, s=s, dspot=dspot, ok=dspot < 2.0, rep=r["rep_err"], nkp=r["n_kp"], h=-pos[2], cpos=pos[:2], last=float(r["frame"] == len(b["F"]) - 1))


def to_goal(cam, g, mirror, origin):
    """pitch (x, y) -> (u, v) m in the goal frame, origin at `origin` (pitch coords)"""
    d = np.asarray(g, float) - origin; return np.stack([cam["s"] * d[..., 0], VSIGN * cam["s"] * d[..., 1] * (-1.0 if mirror else 1.0)], -1)


def warped(b, kp):
    M = b["M"]; P = np.einsum("fij,fkj->fki", M, np.concatenate([kp, np.ones(kp.shape[:2] + (1,), np.float32)], -1)); return P[..., :2] / P[..., 2:3]


def kicker(c, S, mirror, cam, b, kb):
    import kinR
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(np.asarray(vals, np.float32)).ravel(); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; kp, sc = kb["kp"], kb["sc"]; P = warped(b, kp); Pm = np.where((sc >= CT)[..., None], P, np.nan); sg = -1.0 if mirror else 1.0
    G = lambda X, z: to_goal(cam, cam["raw"](X, z), mirror, cam["bg"])
    lg, tk, tp = kinR.legs(P, sc, nF); arr = lambda w, nm: np.array([lg[w][t][nm] for t in range(nF)])
    feet = {(w, nm): G(arr(w, nm), Z[nm]) for w in ("kk", "pl") for nm in ("ank", "toe", "heel")}
    hm, sm, em, nose = np.nanmean(Pm[:, [11, 12]], 1), np.nanmean(Pm[:, [5, 6]], 1), np.nanmean(Pm[:, [3, 4]], 1), Pm[:, 0]
    l3 = slice(max(0, nF - 3), nF); l4 = list(range(max(0, nF - 4), nF)); l6 = list(range(max(0, nF - 6), nF)); near = lambda o: int(np.abs(ts + o).argmin())
    cp = to_goal(cam, cam["cpos"], mirror, cam["bg"]); add("n_cam", [cam["h"], np.hypot(*cp), cp[0] / max(np.hypot(*cp), 1e-6), cp[1] / max(np.hypot(*cp), 1e-6), np.degrees(np.arctan2(cam["h"], np.hypot(*cp))),
                                                                    cam["rep"], cam["nkp"], cam["dspot"], cam["last"]])
    for nm in ("ank", "toe", "heel"): add(f"n_pl_{nm}", feet[("pl", nm)][-1])
    pc = np.nanmean(np.stack([feet[("pl", nm)][l3] for nm in ("ank", "toe", "heel")]), axis=(0, 1)); add("n_pl_cen", pc); add("n_pl_cen_dist", np.hypot(*pc))
    th_ = feet[("pl", "toe")] - feet[("pl", "heel")]; ang = np.arctan2(th_[:, 1], th_[:, 0])
    add("n_pl_dir_last", [np.cos(ang[-1]), np.sin(ang[-1])]); add("n_pl_dir_m3", [np.nanmean(np.cos(ang[l3])), np.nanmean(np.sin(ang[l3]))])
    # plant foot pointing relative to the line from the plant foot to the goal centre (u = 11 m from the ball)
    tg = np.array([11.0, 0.0]) - pc; rel = np.angle(np.exp(1j * (ang[-1] - np.arctan2(tg[1], tg[0])))); add("n_pl_vs_goal", [rel, np.abs(rel)])
    for nm in ("ank", "toe"): add(f"n_kk_{nm}", feet[("kk", nm)][-1])
    kc = np.nanmean(np.stack([feet[("kk", nm)][-1] for nm in ("ank", "toe", "heel")]), 0); add("n_kk_minus_pl", kc - pc)
    for nm in ("ank", "toe"):
        y = feet[("kk", nm)]; vu, vv = slope(ts[l4], y[l4, 0]), slope(ts[l4], y[l4, 1]); add(f"n_kk_{nm}_vel", [vu, vv, np.hypot(vu, vv), np.arctan2(vv, vu) if np.isfinite(vu) else np.nan])
    for nm, (a_, b_), z in (("hipax", (11, 12), Z["hip"]), ("shax", (5, 6), Z["sho"]), ("earax", (3, 4), Z["ear"])):
        V = G(Pm[:, b_], z) - G(Pm[:, a_], z); a2 = 2 * np.arctan2(V[:, 1], V[:, 0]); n = np.hypot(V[:, 0], V[:, 1]); full = np.full(nF, np.nan); fin = np.isfinite(a2)
        if fin.sum() > 1: full[fin] = np.unwrap(a2[fin])
        add(f"n_{nm}_last", [np.cos(a2[-1]), np.sin(a2[-1]), n[-1]]); add(f"n_{nm}_m3", [np.nanmean(np.cos(a2[l3])), np.nanmean(np.sin(a2[l3]))]); add(f"n_{nm}_angvel", slope(ts[l6], full[l6]))
    for nm, S_, z in (("hip", hm, Z["hip"]), ("sho", sm, Z["sho"]), ("nose", nose, Z["nose"])):
        Gp = G(S_, z); D = Gp - Gp[-1]; add(f"n_{nm}_disp", np.concatenate([D[near(o)] for o in (0.15, 0.25, 0.45, 0.8)])); add(f"n_{nm}_pos", Gp[-1]); add(f"n_{nm}_over_plant", Gp[-1] - pc)
        for w in (0.3, 0.8):
            m = np.where(ts >= -w)[0]; add(f"n_{nm}_vel{w}", [slope(ts[m], D[m, 0]), slope(ts[m], D[m, 1])])
    D = G(hm, Z["hip"]); D = D - D[-1]
    for a, b_ in ((0.8, 0.3), (0.3, 0.0)):
        v = D[near(b_)] - D[near(a)]; add(f"n_approach_{a}", [np.arctan2(v[1], v[0]), np.hypot(*v)])
    add("n_foot_sign", (1.0 if META[c]["foot"] == "right" else -1.0) * sg)
    return np.array(f_, np.float32)


def keeper(c, mirror, cam, b, kk):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(np.asarray(vals, np.float32)).ravel(); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    ts = b["ts"] - b["ts"][-1]; P = warped(b, kk["kp"]); P = np.where((kk["sc"] >= CT)[..., None], P, np.nan); near = lambda o: int(np.abs(ts + o).argmin()); ref = near(0.45)
    goal = np.array([52.5 * cam["s"], 0.0]); G = lambda X, z: to_goal(cam, cam["raw"](X, z), mirror, goal)          # origin at the goal centre, u < 0 in front of the line
    for nm, idx, z in (("feet", [15, 16], 0.08), ("hip", [11, 12], 0.95), ("sho", [5, 6], 1.45)):
        S_ = G(np.nanmean(P[:, idx], 1), z)
        add(f"nk_{nm}_pos", [S_[-1, 0], S_[-1, 1], S_[ref, 1]]); add(f"nk_{nm}_disp", [S_[near(o), 1] - S_[ref, 1] for o in (0.0, 0.06, 0.1, 0.15, 0.25)])
        for w in (0.1, 0.2, 0.3, 0.5):
            m = np.where(ts >= -w)[0]; add(f"nk_{nm}_vel{w}", [slope(ts[m], S_[m, 1]), slope(ts[m], S_[m, 0])])
    A = G(P[:, 15], 0.08) - G(P[:, 16], 0.08); add("nk_ank_sep", [np.hypot(*A[-1]), np.hypot(*A[near(0.25)])])
    add("nk_q", [cam["dspot"], cam["rep"]])
    return np.array(f_, np.float32)


def one(args):
    c, S = args; b = load("ballspot", c)
    if b is None: return c, None, None
    cam = camera(c, b)
    if cam is None or not cam["ok"]: return c, "nan", None
    if S in ("kkeep", "kkeepF"):
        k0 = load("kkeep", c); ks = k0 if S == "kkeep" else load(S, c)
        if k0 is None or ks is None: return c, "nan", None
        kk = dict(kp=ks["kp"], sc=ks["sc"]); NAMES.clear(); a = keeper(c, False, cam, b, kk); names = list(NAMES); NAMES.clear(); return c, (a, keeper(c, True, cam, b, kk)), names
    from streams import stream_kb
    kb = stream_kb(c, S)
    if kb is None: return c, "nan", None
    NAMES.clear(); a = kicker(c, S, False, cam, b, kb); names = list(NAMES); NAMES.clear(); return c, (a, kicker(c, S, True, cam, b, kb)), names


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [("kkeep", "keepN_A"), ("kkeepF", "keepN_F")] if sys.argv[1] == "keep" else [(sys.argv[1], f"kinN_{sys.argv[1]}")]
    for S, out in jobs:
        with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, [(c, S) for c in CLIPS], chunksize=20)
        names = next(n for _, r, n in res if n is not None); nan = np.full(len(names), np.nan, np.float32)
        rows = {c: (r if r != "nan" else (nan, nan)) for c, r, _ in res if r is not None}
        pickle.dump(dict(rows=rows, names=names), open(f"{PC}/{out}.pkl", "wb")); X = np.stack([r[0] for r in rows.values()])
        print(f"{out}: rows {len(rows)} | calibrated ok {np.isfinite(X[:, 0]).mean():.3f} (public {np.mean([np.isfinite(rows[c][0][0]) for c in gt['clip'] if c in rows]):.3f}) | dim {len(names)}")
