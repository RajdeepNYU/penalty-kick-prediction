"""Submission from the new parts. Kick = 0.5 * mean(private trees for the given tags) + 0.5 * mean(NN bags EF and G for SEQ, or the v10 NN files if SEQ=v10);
keeper = v10 keeper, or KEEPER=b2 (new-ball keeper features); outcome = goal. usage: [KEEPER=b2] python make_subP.py NAME SEQ treeTag1 [treeTag2 ...]
-> Q4_word_Prime/submission_NAME.csv"""
import os, sys, numpy as np, pandas as pd
from common import *

NAME, SEQ, tags = sys.argv[1], sys.argv[2], sys.argv[3:]; cfg = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]
tmpl = pd.read_csv(D + r"\submission_template.csv"); ck = np.array(["left", "middle", "right"]); cg = np.array(["left", "right", "stands in the middle"])
trees = sum(np.load(f"{PC}/prv_trees_{t}.npy") for t in tags) / len(tags)
if SEQ == "v10": nn = (np.load(f"{W}/nn_private_EF.npy") + np.load(f"{W}/nn_private_G.npy")) / 2
else: sq = SEQ.split("+"); nn = sum(np.load(f"{PC}/prvP_{e}_{h}_st{st}_{q}.npy") for e, h in cfg for st in ("EF", "G") for q in sq) / (10 * len(sq))
Pk = 0.5 * trees + 0.5 * nn
KEEP = os.environ.get("KEEPER", "v10")                                           # v10 keeper, or b2 = keeper features with the new ball (keep_private.py)
Pg = (3 * np.load(f"{W}/pg_private_v7.npy") + np.load(f"{W}/keeper_private_F.npy")) / 5 if KEEP == "v10" else np.load(f"{PC}/pg_private_{KEEP}.npy")
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": ck[Pk.argmax(1)], "goalkeeper_direction": cg[Pg.argmax(1)], "goal_outcome": "goal"})
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique
sub.to_csv(f"{P}/submission_{NAME}.csv", index=False); np.save(f"{PC}/Pk_private_{NAME}.npy", Pk)
print(f"wrote submission_{NAME}.csv | kick {sub.kick_direction.value_counts().to_dict()} | keeper {sub.goalkeeper_direction.value_counts().to_dict()}")
for ref in os.environ.get("COMPARE", ROOT + r"\Q4_work\submission_v10.csv").split(";"):
    if os.path.exists(ref): r = pd.read_csv(ref); print(f"  agreement with {os.path.basename(ref)}: kick {(r.kick_direction == sub.kick_direction).mean():.3f} keeper {(r.goalkeeper_direction == sub.goalkeeper_direction).mean():.3f}")
