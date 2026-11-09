"""Keeper sequence channels for the temporal network (train_seqP.py TARGET=keeper BASE=AF SEQ=kp). Same ball-anchored goal frame as kinkeep
(u = ball -> keeper, v = kicker's left, unit = keeper height; mirror negates v), camera-stabilised with M, ball2 ball. Per F frame (last 12, right-aligned):
hip / feet / shoulder / nose / wrist / knee mids, their displacement from -0.45 s, ankle and knee separation, hip / shoulder axis (2 x angle), keeper box shape,
and from the NBJW camera (when the ball lands within 2 m of a spot) the feet / hip / shoulder lateral position in metres from the goal centre.
Streams: A = RTMW keeper (kkeep), F = ViTPose-L keeper (kkeepF). usage: python keepseq.py  -> cache/seq_A_kp.pkl, cache/seq_F_kp.pkl"""
import pickle, warnings, numpy as np
from common import *
import kinN
warnings.filterwarnings("ignore")

T, CT = 12, 2.0; B2 = pickle.load(open(f"{PC}/ball2.pkl", "rb")); CAM2 = pickle.load(open(f"{PC}/camera2.pkl", "rb"))


def build(c, S, mirror, b, kk, tr, ncam):
    nF = b["nF"]; ts = b["ts"] - b["ts"][-1]; sg = -1.0 if mirror else 1.0
    P = kinN.warped(b, kk["kp"]); P = np.where((kk["sc"] >= CT)[..., None], P, np.nan)
    ball = np.asarray(B2[c]["ball"], float) if B2.get(c) else b["ball_last"]; kb = tr["keep"]; vv = np.where(np.isfinite(kb[:, 0]))[0]
    if len(vv) == 0 or not np.isfinite(ball[0]): return None
    kf = np.array([(kb[vv[-1]][0] + kb[vv[-1]][2]) / 2, kb[vv[-1]][3]]); fv = kf - ball; fn = np.linalg.norm(fv); f = fv / fn if fn > 1e-6 else np.array([1.0, 0.0])
    Lv = np.array([f[1], -f[0]]) * sg; hk = max(float(np.nanmedian(kk["box"][:, 3] - kk["box"][:, 1])), 20.0)
    U = np.stack([(P - ball) @ f / hk, (P - ball) @ Lv / hk], -1); mid = lambda a, b_: np.nanmean(U[:, [a, b_]], 1)
    pts = [mid(11, 12), mid(15, 16), mid(5, 6), U[:, 0], mid(9, 10), mid(13, 14)]; ref = int(np.abs(ts + 0.45).argmin())
    ch = pts + [p - p[ref] for p in pts[:4]]
    ch.append(np.stack([np.hypot(*(U[:, 15] - U[:, 16]).T), np.hypot(*(U[:, 13] - U[:, 14]).T)], 1))
    for a, b_ in ((11, 12), (5, 6)):
        V = U[:, b_] - U[:, a]; a2 = 2 * np.arctan2(V[:, 1], V[:, 0]); ch.append(np.stack([np.cos(a2), np.sin(a2)], 1))
    bx = np.asarray(kk["box"], float); ch.append(np.stack([(bx[:, 2] - bx[:, 0]) / np.maximum(bx[:, 3] - bx[:, 1], 1), (bx[:, 3] - bx[:, 1]) / hk], 1))
    if ncam is not None:
        goal = np.array([52.5 * ncam["s"], 0.0]); G = lambda X, z: kinN.to_goal(ncam, ncam["raw"](X, z), mirror, goal)
        ch.append(np.stack([G(np.nanmean(P[:, [15, 16]], 1), 0.08)[:, 1], G(np.nanmean(P[:, [11, 12]], 1), 0.95)[:, 1], G(np.nanmean(P[:, [5, 6]], 1), 1.45)[:, 1], np.ones(nF)], 1) / np.array([3, 3, 3, 1]))
    else: ch.append(np.zeros((nF, 4)))
    X = np.concatenate(ch, 1).astype(np.float32); fin = np.isfinite(X).mean(1, keepdims=True).astype(np.float32); X = np.clip(np.nan_to_num(X, nan=0.0), -20, 20)
    X = np.concatenate([X, ts[:, None].astype(np.float32), fin, np.ones((nF, 1), np.float32)], 1); seq = np.zeros((T, X.shape[1]), np.float32); n = min(T, nF); seq[-n:] = X[-n:]
    m = META[c]; cam = CAM2.get(c); th, ph = (cam["theta"], np.radians(cam["phi"])) if cam else (np.nan, np.nan)
    st = np.array([(1.0 if m["foot"] == "right" else -1.0) * sg, 1.0 if m["gender"] == "M" else 0.0, float(ncam is not None), 0.0, m["fps"] / 60.0, th / 20.0, np.cos(ph) * sg, np.sin(ph)], np.float32)
    return seq, st


def one(args):
    c, S = args; b = load("ballspot", c); k0 = load("kkeep", c); ks = k0 if S == "A" else load("kkeepF", c); t = tracks()[c]
    if b is None or k0 is None or ks is None or t is None or t["keep"] is None: return c, None
    ncam = kinN.camera(c, b); ncam = ncam if ncam is not None and ncam["ok"] else None
    kk = dict(kp=ks["kp"], sc=ks["sc"], box=k0["box"]); a = build(c, S, False, b, kk, t, ncam); m = build(c, S, True, b, kk, t, ncam)
    return c, (a, m) if a is not None and m is not None else None


if __name__ == "__main__":
    from multiprocessing import Pool
    for S in "AF":
        with Pool(int(os.environ.get("NPOOL", "8"))) as p: res = p.map(one, [(c, S) for c in CLIPS], chunksize=20)
        rows = {c: r for c, r in res if r is not None}; pickle.dump(rows, open(f"{PC}/seq_{S}_kp.pkl", "wb")); r0 = next(iter(rows.values()))
        print(S, "built", len(rows), "shape", r0[0][0].shape, "static", r0[0][1].shape)
