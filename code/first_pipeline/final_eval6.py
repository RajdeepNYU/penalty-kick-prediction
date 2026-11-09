import numpy as np, pickle, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); yk = gt["kick_direction"].values; yg = gt["goalkeeper_direction"].values; yo = gt["goal_outcome"].values
cl = np.array(["left", "middle", "right"]); acc = lambda P: (cl[P.argmax(1)] == yk).mean()
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); trees = (M["hgb259"] + M["extratrees"]) / 2
cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
nn = sum(np.load(f"{C}/oof_seq_{e}_{h}_k0_s1_conv_b1.npy") for e, h in cfg) / len(cfg)
Pk = 0.5 * trees + 0.5 * nn; pred = cl[Pk.argmax(1)]; lr = yk != "middle"
print(f"trees {acc(trees):.3f} | NN bag(5 configs) {acc(nn):.3f} | FINAL kick (0.5 trees + 0.5 NN) {acc(Pk):.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f}")
pm_ = pred == "middle"; print(f"predicted middle: {pm_.sum()} clips, precision {(yk[pm_] == 'middle').mean() if pm_.any() else float('nan'):.2f} (base rate 0.14), recall {(pred[yk == 'middle'] == 'middle').mean():.2f}")
print("NN alone predicted middle:", (cl[nn.argmax(1)] == "middle").sum(), "precision %.2f" % (yk[cl[nn.argmax(1)] == "middle"] == "middle").mean())
KL = pickle.load(open(f"{C}/oof_keep_lab.pkl", "rb")); Pg = (KL["new_hgb"] + KL["new_et"] + KL["cnn"]) / 3; cg = np.array(["left", "right", "stands in the middle"]); ag = (cg[Pg.argmax(1)] == yg).mean()
go = (yo == "goal").mean(); print(f"keeper {ag:.3f} | outcome {go:.3f} | ESTIMATED CV SCORE v6: {0.6 * acc(Pk) + 0.25 * ag + 0.15 * go:.4f}  (v5 was 0.7238, LB 0.7225)")
conf = Pk.max(1)
for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)): m = (conf >= lo) & (conf < hi); print(f"   confidence [{lo},{hi}) n={m.sum()} acc {(pred[m] == yk[m]).mean():.3f}")
