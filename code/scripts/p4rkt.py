"""P4rkT source-grouped out-of-fold kick probabilities on the public clips (left, middle, right), same construction as kick_metrics.py / mid_gain.py. Import: from p4rkt import Pk"""
import pickle, numpy as np
from common import *
from blend_eval import trP, bagP

KC = P + r"\cache_kf"; _pub = list(gt["clip"])
_KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb"))); mk = np.array([c in _KF for c in _pub])
_B3 = pickle.load(open(f"{PC}/ball3.pkl", "rb")); mb = np.array([bool(_B3.get(c)) and _B3[c].get("method") == "nbjw_spot" for c in _pub])
T2 = sum(trP(f"{S}_b2R") for S in "AFG") / 3; N2 = sum(bagP(st, q) for st in ("EF", "G") for q in ("b2", "b2RM")) / 4
Tb = sum(trP(f"{S}_b3R") for S in "AFG") / 3; Nb = sum(bagP(st, q) for st in ("EF", "G") for q in ("b2_b3", "b2RM_b3")) / 4
Tk = sum((np.load(f"{KC}/oof_hgb_{S}_b2R.npy") + np.load(f"{KC}/oof_et_{S}_b2R.npy")) / 2 for S in "AFG") / 3
Pk = np.where(mk[:, None], Tk, np.where(mb[:, None], 0.5 * Tb + 0.5 * Nb, 0.5 * T2 + 0.5 * N2)); Pk = Pk / Pk.sum(1, keepdims=True)
