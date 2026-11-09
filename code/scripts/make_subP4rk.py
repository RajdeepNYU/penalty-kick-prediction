"""P4b3k + kicker re-pick (hybrid). Kick: on the private clips whose kicker kfix.py re-picked, 0.5 * mean(prv_trees {A,F,G}_b2R) + 0.5 * mean(NN full-mode
bags b2, b2RM on EF and G), all from cache_kf (pipeline34 / pipeline35); every other clip keeps P4b3k's kick (Pk_private_P4b3.npy). Keeper: P4b3k's.
ARM=T: re-pick trees only on those clips (chosen after the CV, kfix_eval.py). -> submission_P4rk.csv / submission_P4rkT.csv (not submitted)"""
import pickle, numpy as np, pandas as pd
from common import *

KC = P + r"\cache_kf"; cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; ck = np.array(["left", "middle", "right"])
KF = set(pickle.load(open(f"{PC}/kfix_clips.pkl", "rb"))); mv = np.array([c in KF for c in pm["clip"]])
T = sum(np.load(f"{KC}/prv_trees_{S}_b2R.npy") for S in "AFG") / 3
N = sum(np.load(f"{KC}/prvP_{e}_{h}_st{st}_{q}.npy") for e, h in cfg for st in ("EF", "G") for q in ("b2", "b2RM")) / 20
Pb = np.load(f"{PC}/Pk_private_P4b3.npy"); base = pd.read_csv(f"{P}/submission_P4b3k.csv"); ref = pd.read_csv(f"{P}/submission_P4.csv")
assert list(base["clip"]) == list(pm["clip"]) and (ck[Pb.argmax(1)] == base["kick_direction"].values).all()
ARM = os.environ.get("ARM", ""); NAME = "P4rk" + ARM
Pk = np.where(mv[:, None], T if ARM == "T" else 0.5 * T + 0.5 * N, Pb); sub = base.assign(kick_direction=ck[Pk.argmax(1)]); sub.to_csv(f"{P}/submission_{NAME}.csv", index=False)
np.save(f"{PC}/Pk_private_{NAME}.npy", Pk); chg = sub.kick_direction != base.kick_direction
print(f"wrote submission_{NAME}.csv | re-picked private clips {mv.sum()} | kick changed on {int(chg.sum())} of them vs P4b3k "
      f"({base.kick_direction[chg].str[:1].add('->').add(sub.kick_direction[chg].str[:1]).value_counts().to_dict()}) | vs P4: kick {int((sub.kick_direction != ref.kick_direction).sum())}, "
      f"keeper {int((sub.goalkeeper_direction != ref.goalkeeper_direction).sum())}")
