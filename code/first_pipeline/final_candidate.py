import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk, yg, yo = gt["kick_direction"].values, gt["goalkeeper_direction"].values, gt["goal_outcome"].values; cl = np.array(["left", "middle", "right"]); cg = np.array(["left", "right", "stands in the middle"])
ak = lambda P: (cl[P.argmax(1)] == yk).mean(); ag = lambda P: (cg[P.argmax(1)] == yg).mean(); lr = yk != "middle"; go = (yo == "goal").mean(); cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tr = lambda n: (np.load(f"{C}/base_hgb{n}.npy") + np.load(f"{C}/base_et{n}.npy")) / 2; bag = lambda st: sum(np.load(f"{C}/oof_seq2_{e}_{h}_b1_kc0_ax0_j0_st{st}.npy") for e, h in cfg) / 5
trees3 = (tr("") + tr("_F") + tr("_G")) / 3; nn = (bag("EF") + bag("G")) / 2; Pk = 0.5 * trees3 + 0.5 * nn
print(f"v9 recipe  (trees RTMW+L, NN A+B+EF):                 kick {ak(0.5 * (tr('') + tr('_F')) / 2 + 0.5 * bag('EF')):.3f}")
print(f"v10 recipe (trees RTMW+L+H, NN bags EF + G averaged):  kick {ak(Pk):.3f} (L/R-only {(cl[Pk.argmax(1)][lr] == yk[lr]).mean():.3f})")
kr, kf = np.load(f"{C}/keep_oof_rtmw.npy"), np.load(f"{C}/keep_oof_F.npy"); cnn = np.load(f"{C}/oof_gk_resnet34.a1_in1k_0.npy")[:, [0, 2, 1]]
Pg9 = (2 * kr + cnn) / 3; Pg10 = (2 * kr + 2 * kf + cnn) / 5
print(f"keeper: v9 blend {ag(Pg9):.3f} -> v10 blend (RTMW trees + ViTPose-L trees + CNN, 2:2:1) {ag(Pg10):.3f}; equal thirds {ag((kr + kf + cnn) / 3):.3f}")
for nm, k_, g_ in (("v9", 0.5 * (tr('') + tr('_F')) / 2 + 0.5 * bag('EF'), Pg9), ("v10", Pk, Pg10)): print(f"  estimated CV score {nm}: {0.6 * ak(k_) + 0.25 * ag(g_) + 0.15 * go:.4f}")
