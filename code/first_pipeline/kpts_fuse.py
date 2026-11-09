"""Stream M: per-frame fusion of the independent pose measurements (RTMW, ViTPose base/large/huge) into one lower-noise skeleton.
COCO joints 0-16 = confidence-weighted mean over streams (weights x scale-normalised; ViTPose scores are on the RTMW scale already); RTMW-only slots (toes/heels, hands, face) copied from RTMW. Writes cache/kptsM."""
import pickle, os, numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"; os.makedirs(f"{C}/kptsM", exist_ok=True)
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv"); W = {"kpts": 1.0, "kptsE": 1.0, "kptsF": 1.5, "kptsG": 1.5}; n = 0
for c in list(gt["clip"]) + list(pm["clip"]):
    S = {k: pickle.load(open(f"{C}/{k}/{c}.pkl", "rb")) for k in W}
    if any(v is None for v in S.values()): pickle.dump(None, open(f"{C}/kptsM/{c}.pkl", "wb")); continue
    kp = S["kpts"]["kp"].copy(); sc = S["kpts"]["sc"].copy(); num = 0; den = 0; ss = 0
    for k, w in W.items():
        s = np.clip(S[k]["sc"][:, :17], 0.05, None) * w; num = num + S[k]["kp"][:, :17] * s[..., None]; den = den + s; ss = ss + S[k]["sc"][:, :17] * w
    kp[:, :17] = num / den[..., None]; sc[:, :17] = ss / sum(W.values())
    pickle.dump(dict(kp=kp, sc=sc, box=S["kpts"]["box"]), open(f"{C}/kptsM/{c}.pkl", "wb")); n += 1
print("fused", n)
