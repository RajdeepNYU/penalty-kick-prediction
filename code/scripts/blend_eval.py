"""CV score of the v10 recipe with parts swapped for the new ones. Kick: 0.5 * mean(trees A, F, G) + 0.5 * mean(NN bag EF, NN bag G);
keeper: v10 blend (unchanged unless given); outcome: always 'goal'. Score = 0.6 kick + 0.25 keeper + 0.15 outcome. usage: python blend_eval.py"""
import os, numpy as np
from common import *
from cvlib import acc, lr_acc, yk

cg = np.array(["left", "right", "stands in the middle"]); yg = gt["goalkeeper_direction"].values; go = (gt["goal_outcome"].values == "goal").mean()
ag = lambda P: float((cg[P.argmax(1)] == yg).mean()); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
trW = lambda n: (np.load(f"{W}/base_hgb{n}.npy") + np.load(f"{W}/base_et{n}.npy")) / 2
trP = lambda t: (np.load(f"{PC}/oof_hgb_{t}.npy") + np.load(f"{PC}/oof_et_{t}.npy")) / 2
bagW = lambda st: sum(np.load(f"{W}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
bagP = lambda st, sq: sum(np.load(f"{PC}/oofP_{e}_{h}_st{st}_{sq}.npy") for e, h in cfg) / 5
Pg10 = (2 * np.load(f"{W}/keep_oof_rtmw.npy") + 2 * np.load(f"{W}/keep_oof_F.npy") + np.load(f"{W}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]) / 5
score = lambda Pk, Pg=Pg10: 0.6 * acc(Pk) + 0.25 * ag(Pg) + 0.15 * go
have = lambda f: os.path.exists(f"{PC}/{f}")


def report(name, trees, nn):
    Pk = 0.5 * trees + 0.5 * nn
    print(f"{name:44s} trees {acc(trees):.4f} | nn {acc(nn):.4f} | kick {acc(Pk):.4f} (L/R {lr_acc(Pk):.4f}) | CV score {score(Pk):.4f}")
    return Pk


if __name__ == "__main__":
    T10 = (trW("") + trW("_F") + trW("_G")) / 3; N10 = (bagW("EF") + bagW("G")) / 2
    report("v10 (reference)", T10, N10)
    for tag in ("b2", "b2R"):
        if all(have(f"oof_hgb_{s}_{tag}.npy") for s in "AFG"):
            T = (trP(f"A_{tag}") + trP(f"F_{tag}") + trP(f"G_{tag}")) / 3; report(f"trees {tag} + v10 NN", T, N10)
            for sq in ("b2", "b2R"):
                if all(have(f"oofP_{e}_{h}_st{st}_{sq}.npy") for e, h in cfg for st in ("EF", "G")):
                    report(f"trees {tag} + NN {sq}", T, (bagP("EF", sq) + bagP("G", sq)) / 2)
    print(f"keeper v10 {ag(Pg10):.4f} | outcome 'goal' {go:.4f}")
