"""Sequence channels (Q4_work/seqfeat.build, unchanged) with the new ball (ball2), optionally with the rectified per-frame channels of kinR appended.
usage: python seq2b.py STREAM [R|-] [sK]   -> cache/seq_{STREAM}_b2[R][_sK].pkl  (sK: window ending K native frames earlier, public clips only)  rows[c] = ((X [T,C], static), (Xm, static_m)), same layout as Q4_work/cache/seq*.pkl"""
import sys, pickle, warnings, numpy as np
from multiprocessing import Pool
from common import *
from streams import stream_kb
warnings.filterwarnings("ignore")
import seqfeat

EXTRA = {"R3": "kin3R", "RM": "kinMB", "RM2": "kinMB2", "RMR": "kinMBR", "RMU": "kinMBU"}.get(sys.argv[2] if len(sys.argv) > 2 else "")   # also append 3D goal-frame channels: R3 = RTMW3D (kin3R.py), RM = MotionBERT-Lite (kinMB.py), RMR = + its representation (kinMBR.py); the same for every stream
S = sys.argv[1]; RCH = len(sys.argv) > 2 and (sys.argv[2] == "R" or EXTRA is not None); SH = int(sys.argv[3][1:]) if len(sys.argv) > 3 else 0     # e.g. "s1": window ends 1 native frame earlier
R3 = EXTRA is not None
BALLF = os.environ.get("BALLF", "ball2"); SEQTAG = {"ball2": "", "ball3": "_b3", "ball3b": "_b3b"}[BALLF]      # BALLF=ball3 (mkball3.py; run with CAMF=camera2_ball3) -> seq_{S}_b2..._b3.pkl
SEQTAG += {"camera2_cam3": "_c3"}.get(os.environ.get("CAMF", "camera2"), "")                                    # CAMF=camera2_cam3 (camera3.py), ball2 -> seq_{S}_b2RM_c3.pkl
B2 = pickle.load(open(f"{PC}/{BALLF}.pkl", "rb")); K3 = pickle.load(open(f"{PC}/{EXTRA}{SEQTAG}.pkl", "rb"))["seq"] if R3 else None; K3D = next(iter(K3.values()))[0].shape[1] if R3 else 0


def one(c):
    b = load("ballspot", c); kb = stream_kb(c, S); t = tracks()[c]
    if b is None or kb is None or t is None or t["keep"] is None: return c, None
    b = dict(b)
    if B2.get(c) is not None: b["ball_last"] = np.asarray(B2[c]["ball"], float)
    if SH:
        keep = np.where(b["ts"] <= b["ts"][-1] - (SH - 0.5) / b["fps_native"])[0]
        if len(keep) < 6: return c, None
        nF0 = b["nF"]; b.update(F=np.asarray(b["F"])[keep], ts=b["ts"][keep], M=b["M"][keep], nF=len(keep))
        kb = dict(kb, kp=kb["kp"][keep], sc=kb["sc"][keep], box=np.asarray(kb["box"])[keep] if len(kb["box"]) == nF0 else kb["box"])
    a = seqfeat.build(b, kb, t, META[c], False); m = seqfeat.build(b, kb, t, META[c], True)
    if a is None or m is None: return c, None
    if RCH:
        import kinR
        cam = kinR.CAM.get(c)
        if cam is None: return c, None
        out = []
        for mir, base in ((False, a), (True, m)):
            _, seqR, nst = kinR.build(c, S, mir, cam, b, kb, kb["box"]); X = base[0]
            ex = [K3[c][int(mir)] if c in K3 else np.zeros((len(X), K3D), np.float32)] if R3 else []
            out.append((np.concatenate([X[:, :-3], seqR[:, :-3]] + ex + [X[:, -3:]], 1), np.r_[base[1], nst[3:]].astype(np.float32)))
        return c, tuple(out)
    return c, (a, m)


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPOOL", "12"))) as p: res = p.map(one, list(gt["clip"]) if SH else CLIPS, chunksize=20)
    rows = {c: r for c, r in res if r is not None}
    pickle.dump(rows, open(f"{PC}/seq_{S}_b2{sys.argv[2] if R3 else 'R' if RCH else ''}{f'_s{SH}' if SH else ''}{SEQTAG}.pkl", "wb")); r0 = next(iter(rows.values()))
    print(S, "built", len(rows), "shape", r0[0][0].shape, "static", r0[0][1].shape)
