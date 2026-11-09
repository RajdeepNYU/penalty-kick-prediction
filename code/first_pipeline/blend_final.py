import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; yg = gt["goalkeeper_direction"].values; yo = gt["goal_outcome"].values; cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean(); lr = yk != "middle"
LR = lambda P: (cl[P.argmax(1)][lr] == yk[lr]).mean(); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tA = (np.load(f"{C}/base_hgb.npy") + np.load(f"{C}/base_et.npy")) / 2; tF = (np.load(f"{C}/base_hgb_F.npy") + np.load(f"{C}/base_et_F.npy")) / 2; tB = (tA + tF) / 2
nnE = sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_stEF.npy") for e, h in cfg) / 5; nnF = sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_stF.npy") for e, h in cfg) / 5; nn = (nnE + nnF) / 2
print(f"trees RTMW {acc(tA):.3f} | trees ViTPose-L {acc(tF):.3f} | trees both {acc(tB):.3f} | NN (A+B+EF) {acc(nnE):.3f} | NN (A+B+F) {acc(nnF):.3f} | NN both bags {acc(nn):.3f}")
for name, T in (("trees RTMW only", tA), ("trees both", tB)):
    for nname, N in (("NN A+B+EF", nnE), ("NN both bags", nn)):
        print(f"  {name:16s} + {nname:13s}: " + " | ".join(f"w_nn={w}: {acc((1 - w) * T + w * N):.3f}" for w in (0.4, 0.5, 0.6)) + f"   (L/R-only at 0.5: {LR(0.5 * T + 0.5 * N):.3f})")
Pk = 0.5 * tB + 0.5 * nn; Pg = np.load(f"{C}/new_Pg.npy"); ag = (np.array(["left", "right", "stands in the middle"])[Pg.argmax(1)] == yg).mean(); go = (yo == "goal").mean()
print(f"\nFINAL CANDIDATE: kick {acc(Pk):.3f} (L/R-only {LR(Pk):.3f}), keeper {ag:.3f}, outcome {go:.3f} -> estimated CV score {0.6 * acc(Pk) + 0.25 * ag + 0.15 * go:.4f}   (v7: kick .697, score .7538)")
