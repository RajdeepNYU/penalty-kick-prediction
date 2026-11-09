"""P4rkD + the adopted 4 Nov ideas (RULES_nov4.md, decisions in logs/nov4/eval.txt). Kick = blend of base (P4rkT) and pixel (P4rkD's three ViTPose+ base crops); keeper = P4rkT's. Flags:
--g15  base trees for the private clips at 15 fps from the 25 fps grid rows (cache_g), trees trained on the original rows (idea 1c)
--cam  base trees and pixel camera terms from the corrected camera on the private clips with sin phi < 0 (idea 3a)
--dc   pixel model on CLAHE crops (idea 4)
--mix  pixel model trained on original + degraded crops for the private clips at 640x360 (idea 1b)
--geo  geometric blend (idea 2a); --mid  P(middle) x 0.8 (idea 2b)
Re-picked / ball3-moved clips keep P4rkT's base. No flags reproduces P4rkD (checked). Writes only the local file; submitting is a separate step.
usage: python make_subN4.py NAME [flags] -> submission_{NAME}.csv, cache/Pk_private_{NAME}.npy"""
import sys, pickle, numpy as np, pandas as pd
from common import *

NAME, FL = sys.argv[1], set(a[2:] for a in sys.argv[2:]); cl = np.array(["left", "middle", "right"]); prv = list(pm["clip"]); NV = f"{PC}/nov4"; cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb"))); B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb")); oth = np.array([c in KF or (bool(B3.get(c)) and B3[c].get("method") == "nbjw_spot") for c in prv])
Pb = np.load(f"{PC}/Pk_private_P4rkT.npy"); base = pd.read_csv(f"{P}/submission_P4rkT.csv"); assert list(base["clip"]) == prv
N = sum(np.load(f"{PC}/prvP_{e}_{h}_st{st}_{q}.npy") for e, h in cfg for st in ("EF", "G") for q in ("b2", "b2RM")) / 20; F = np.load(f"{PC}/Pp_e_body_hm_vpbaseD+e_body_tor_vpbaseD+e_legs12_vpbaseD_512_0.001.npy", allow_pickle=True).item()["P"]
f15 = pm["fps"].values < 20; lr = pm["resolution"].values == "640x360"; CF = pickle.load(open(f"{PC}/camera2f.pkl", "rb")); fc = np.array([bool(CF.get(c)) and "phi_old" in CF[c] for c in prv]); Bk = Pb.copy(); log = []
if "cam" in FL:
    T = sum(np.load(f"{NV}/prv_trees_{S}.npz")["cam"] for S in "AFG") / 3; m = fc & ~oth; Bk[m] = .5 * T[m] + .5 * N[m]; log.append(f"cam base on {int(m.sum())} clips")
if "g15" in FL:
    T = sum(np.load(f"{NV}/prv_trees_{S}.npz")["grid"] for S in "AFG") / 3; m = f15 & ~oth; Bk[m] = .5 * T[m] + .5 * N[m]; log.append(f"grid base on {int(m.sum())} clips at 15 fps")
if "dc" in FL: F = np.load(f"{NV}/prv_px_DC.npy"); log.append("CLAHE pixel on all clips")
if "cam" in FL and "dc" not in FL: Fc = np.load(f"{NV}/prv_px_cam.npy"); F = np.where(fc[:, None], Fc, F); log.append(f"cam pixel on {int(fc.sum())} clips")
if "mix" in FL: Fm = np.load(f"{NV}/prv_px_mix.npy"); F = np.where(lr[:, None], Fm, F); log.append(f"mix pixel on {int(lr.sum())} clips at 640x360")
if "geo" in FL: G = np.exp(.5 * np.log(Bk + 1e-4) + .5 * np.log(F + 1e-4)); Pk = G / G.sum(1, keepdims=True); log.append("geometric blend")
else: Pk = .5 * Bk + .5 * F
if "mid" in FL: Pk = Pk * np.array([1, .8, 1]); Pk = Pk / Pk.sum(1, keepdims=True); log.append("middle x0.8")
D = np.load(f"{PC}/Pk_private_P4rkD.npy")
if not FL: assert np.allclose(Pk, D), "no flags must reproduce P4rkD"
sub = base.assign(kick_direction=cl[Pk.argmax(1)]); sub.to_csv(f"{P}/submission_{NAME}.csv", index=False); np.save(f"{PC}/Pk_private_{NAME}.npy", Pk); d = cl[D.argmax(1)]; chg = sub.kick_direction.values != d
print(f"wrote submission_{NAME}.csv | {'; '.join(log) or 'P4rkD as is'} | kick changed vs P4rkD on {int(chg.sum())} of {len(sub)} ({pd.Series([a[0] + '->' + b[0] for a, b in zip(d[chg], sub.kick_direction.values[chg])]).value_counts().to_dict()}) | by group: 15 fps {int((chg & f15).sum())}, 640x360 {int((chg & lr).sum())}, camera-flagged {int((chg & fc).sum())} | counts {sub.kick_direction.value_counts().to_dict()}")
