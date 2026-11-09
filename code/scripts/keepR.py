"""Keeper features in the GOAL PLANE. The keeper stays on the goal line, so his motion lies in the vertical plane spanned by a_v (1 m along the goal line, to the
kicker's left) and a_z (1 m up) of the per-clip camera (camera2.pkl). Image displacements are decomposed on [a_v a_z] -> (lateral, up) in metres.
This 2x2 system is well conditioned for every view, unlike the ground decomposition on [a_u a_v] (nearly parallel in the image for low cameras).
Keypoints: RTMW (kkeep, stream A) or ViTPose-L (kkeepF, stream F), stabilised as in kinkeep. Mirror negates the lateral axis.
usage: python keepR.py  -> cache/keepR_{A,F}.pkl rows {clip: (static, static_mirror)}"""
import sys, pickle, warnings, numpy as np
from multiprocessing import Pool
from common import *
warnings.filterwarnings("ignore")
from kin2 import slope

CT = 2.0; NAMES = []
CAMF = os.environ.get("CAMF", "camera2"); CAM = pickle.load(open(f"{PC}/{CAMF}.pkl", "rb")); OSFX = "_b3" if CAMF == "camera2_ball3" else ""   # CAMF=camera2_ball3 -> keepR_{A,F}_b3.pkl


def build(b, kk, cam, mirror):
    f_ = []
    def add(name, vals):
        vals = np.atleast_1d(np.asarray(vals, np.float32)).ravel(); f_.extend(vals.tolist()); NAMES.extend([f"{name}{i}" if len(vals) > 1 else name for i in range(len(vals))])
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; sc = kk["sc"]; M = b["M"]; sg = -1.0 if mirror else 1.0
    P = np.einsum("fij,fkj->fki", M, np.concatenate([kk["kp"], np.ones(kk["kp"].shape[:2] + (1,), np.float32)], -1)); P = P[..., :2] / P[..., 2:3]
    P = np.where((sc >= CT)[..., None], P, np.nan)
    Bi = np.linalg.inv(np.c_[cam["A"][:, 1], cam["az"]])
    GP = lambda V: (np.asarray(V, float) @ Bi.T) * np.array([sg, 1.0])                     # image vector -> (lateral to kicker's left, up) m in the goal plane
    mid = lambda ids: np.nanmean(P[:, ids], axis=1)
    pts = {"hip": mid([11, 12]), "sho": mid([5, 6]), "head": P[:, 0], "feet": mid([15, 16]), "wri": mid([9, 10]), "knee": mid([13, 14])}
    near = lambda o: int(np.abs(ts + o).argmin()); ref = near(0.45)
    for nm, S in pts.items():
        D = GP(S - S[ref]); add(f"g_{nm}_disp", np.concatenate([D[near(o)] for o in (0.0, 0.03, 0.06, 0.1, 0.15, 0.25)]))
        for w in (0.1, 0.2, 0.3, 0.5):
            m = np.where(ts >= -w)[0]; add(f"g_{nm}_vel{w}", [slope(ts[m], D[m, 0]), slope(ts[m], D[m, 1])])
        m1, m2 = np.where((ts >= -0.3) & (ts < -0.12))[0], np.where(ts >= -0.12)[0]
        add(f"g_{nm}_acc", slope(ts[m2], D[m2, 0]) - slope(ts[m1], D[m1, 0]) if len(m1) > 1 and len(m2) > 1 else np.nan)
    # posture in the goal plane (lateral lean of trunk / legs, arm and foot spread, centre of mass over the feet)
    for nm, (a_, b_) in (("trunk", ("hip", "sho")), ("head", ("sho", "head")), ("leg", ("feet", "hip")), ("thigh", ("knee", "hip"))):
        V = GP(pts[b_] - pts[a_]); lean = np.arctan2(V[:, 0], np.maximum(np.abs(V[:, 1]), 1e-3) * np.sign(V[:, 1] + 1e-9)); m = np.where(ts >= -0.3)[0]
        add(f"g_lean_{nm}", [lean[-1], np.nanmean(lean[-3:]), lean[near(0.25)], slope(ts[m], lean[m])])
    for nm, (a_, b_) in (("ank", (15, 16)), ("wri", (9, 10)), ("knee", (13, 14))):
        V = GP(P[:, b_] - P[:, a_]); add(f"g_{nm}_spread", [np.abs(V[-1, 0]), np.abs(V[near(0.25), 0]), V[-1, 1] * 0 + np.abs(V[-1, 1])])
    cm = GP(pts["hip"] - pts["feet"]); add("g_com_over_feet", [cm[-1, 0], cm[near(0.1), 0], cm[near(0.25), 0]])
    wr = [GP(P[:, j] - pts["sho"]) for j in (9, 10)]; add("g_wrists", [np.nanmax([wr[0][-1, 1], wr[1][-1, 1]]), np.nansum([wr[0][-1, 0], wr[1][-1, 0]])])
    # keeper position on the goal line relative to the goal centre (posts only)
    if cam["posts"] and not cam.get("posts_rejected", 0):
        o = GP(pts["hip"] - cam["Gc"]); of = GP(pts["feet"] - cam["Gc"]); add("g_off_goal", [o[-1, 0], o[ref, 0], of[-1, 0], of[ref, 0], of[-1, 1]])
    else: add("g_off_goal", [np.nan] * 5)
    add("g_cam", [cam["theta"], np.cos(np.radians(cam["phi"])) * sg, np.sin(np.radians(cam["phi"]))])
    return np.array(f_, np.float32)


def one(args):
    c, S = args
    b = load("ballspot", c); k0 = load("kkeep", c); ks = k0 if S == "A" else load("kkeepF", c); cam = CAM.get(c)
    if b is None or k0 is None or ks is None or cam is None: return c, None, None
    kk = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"])
    NAMES.clear(); a = build(b, kk, cam, False); names = list(NAMES); NAMES.clear(); m = build(b, kk, cam, True)
    return c, (a, m), names


if __name__ == "__main__":
    for S in "AF":
        with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, [(c, S) for c in CLIPS], chunksize=20)
        rows = {c: r for c, r, _ in res if r is not None}; names = next(n for _, r, n in res if r is not None)
        pickle.dump(dict(rows=rows, names=names), open(f"{PC}/keepR_{S}{OSFX}.pkl", "wb"))
        X = np.stack([r[0] for r in rows.values()]); print(f"keepR {S}: built {len(rows)} | dim {len(names)} | NaN frac {np.isnan(X).mean():.3f}")
