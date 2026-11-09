"""P4 + ball3 (hybrid) + keeper NN. Kick: on the private clips whose ball mkball3.py moved to the NBJW-projected spot, 0.5 * mean(prv_trees {A,F,G}_b3R)
+ 0.5 * mean(NN full-mode bags b2_b3, b2RM_b3 on EF and G); every other clip keeps P4's kick (Pk_private_P4.npy). Keeper: submission_P4k.csv (P4 keeper + keeper NN).
ARM=b3b: the same with ball3b (mkball3.py DSPOT=0.5). -> submission_P4b3k.csv / submission_P4b3bk.csv (not submitted)"""
import pickle, numpy as np, pandas as pd
from common import *

ARM = os.environ.get("ARM", "b3"); BF = {"b3": "ball3", "b3b": "ball3b"}[ARM]
cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; ck = np.array(["left", "middle", "right"])
B3 = pickle.load(open(f"{PC}/{BF}.pkl", "rb")); mv = np.array([bool(B3.get(c)) and B3[c].get("method") == "nbjw_spot" for c in pm["clip"]])
T = sum(np.load(f"{PC}/prv_trees_{S}_{ARM}R.npy") for S in "AFG") / 3
N = sum(np.load(f"{PC}/prvP_{e}_{h}_st{st}_{q}.npy") for e, h in cfg for st in ("EF", "G") for q in (f"b2_{ARM}", f"b2RM_{ARM}")) / 20
P4 = np.load(f"{PC}/Pk_private_P4.npy"); base = pd.read_csv(f"{P}/submission_P4k.csv"); ref = pd.read_csv(f"{P}/submission_P4.csv")
assert list(base["clip"]) == list(pm["clip"]) and (ck[P4.argmax(1)] == ref["kick_direction"].values).all()
Pk = np.where(mv[:, None], 0.5 * T + 0.5 * N, P4); sub = base.assign(kick_direction=ck[Pk.argmax(1)]); sub.to_csv(f"{P}/submission_P4{ARM}k.csv", index=False)
np.save(f"{PC}/Pk_private_P4{ARM}.npy", Pk); chg = sub.kick_direction != ref.kick_direction
full = ck[(0.5 * T + 0.5 * N).argmax(1)] != ref.kick_direction
print(f"wrote submission_P4{ARM}k.csv | moved private clips {mv.sum()} | kick changed on {int(chg.sum())} of them vs P4 "
      f"({ref.kick_direction[chg].str[:1].add('->').add(sub.kick_direction[chg].str[:1]).value_counts().to_dict()}) | a full swap would change {int(full.sum())} | "
      f"keeper vs P4 changed {int((sub.goalkeeper_direction != ref.goalkeeper_direction).sum())}")
