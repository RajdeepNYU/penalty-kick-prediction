"""Shared by the overfitting tests (ovf_*.py): P4rkD out-of-fold on both seed sets, keeper, per-clip scores, leaderboard facts. Read-only.
Leaderboards (README section 5): both are computed on the 900 private clips, a fixed random 40% (360 clips) is public, the other 60% (540) decides the final rank. The 1300 labelled clips are on neither board."""
import numpy as np, pandas as pd
from common import *
from cvlib import cl, yk, grp
from p4rkt import Pk, mk, mb

N_PUB, N_PRV, N_TEST = 360, 540, 900; PUB_D = 0.83611; PUB_KICK, PUB_KEEP = 289, 329     # P4rkD public = (7 * 289 + 3 * 329) / 3600 = 0.83611
cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; oth = mk | mb; side = yk != "middle"; SETS = ("", "_sd345"); pub, prv = list(gt["clip"]), list(pm["clip"])
cg = np.array(["left", "right", "stands in the middle"]); yg = gt["goalkeeper_direction"].values; src = gt["source_id"].values; us = np.unique(src); ix = {u: np.where(src == u)[0] for u in us}


def base(SF=""):
    T = sum((np.load(f"{PC}/oof_hgb_{S}_b2R{SF}.npy") + np.load(f"{PC}/oof_et_{S}_b2R{SF}.npy")) / 2 for S in "AFG") / 3
    N = sum(np.load(f"{PC}/oofP_{e}_{h}_st{st}_{q}{SF}.npy") for e, h in cfg for st in ("EF", "G") for q in ("b2", "b2RM")) / 20
    return np.where(oth[:, None], Pk, 0.5 * T + 0.5 * N)


B = {s: base(s) for s in SETS}; F = {s: np.load(f"{PC}/oof_px_cat3D{s}.npy") for s in SETS}; Q = {s: 0.5 * B[s] + 0.5 * F[s] for s in SETS}; Qm = (Q[""] + Q["_sd345"]) / 2
ws = dict(A_b2=2, F_b2=2, A_G=1, F_G=1, CNN=1, NNk=1); Pg = sum(w * (np.load(f"{W}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]] if m == "CNN" else np.load(f"{PC}/keep_oof_{m}.npy")) for m, w in ws.items()) / sum(ws.values())
okq = lambda P: (cl[P.argmax(1)] == yk).astype(float); okg = (cg[Pg.argmax(1)] == yg).astype(float); okk = {s: okq(Q[s]) for s in SETS}; okm = (okk[""] + okk["_sd345"]) / 2; score = {s: 0.7 * okk[s] + 0.3 * okg for s in SETS}
metric = lambda k, g: 0.7 * k + 0.3 * g; CV_KICK, CV_KEEP = float(okm.mean()), float(okg.mean()); CV = metric(CV_KICK, CV_KEEP)
Ptest = np.load(f"{PC}/Pk_private_P4rkD.npy"); Pg_test = np.load(f"{PC}/pg_private_b2GNNk.npy")
