"""N4 + what RULES_nov6.md adopted (decisions in logs/nov6/judge.txt): idea 3 (G3), keeper dip timing and "committed yet?" flags (keepD.py) appended to the four keeper
tree members. Kick column = N4's, unchanged (K2 and Tm were not adopted). Keeper = (7 * b2GD + keeper network) / 8, the same construction as N4's (7 * b2G + NNk) / 8.
Flags: --g3 new keeper (cache/nov6/pg_private_b2GD.npy from nov6_private.py keeper). No flags rebuilds submission_N4.csv exactly (asserted).
Writes only the local file; submitting is a separate step. usage: python make_subN6.py NAME [--g3] -> submission_{NAME}.csv, cache/pg_private_{NAME}.npy"""
import sys, numpy as np, pandas as pd
from common import *

NAME, FL = sys.argv[1], set(a[2:] for a in sys.argv[2:]); cg = np.array(["left", "right", "stands in the middle"])
n4 = pd.read_csv(f"{P}/submission_N4.csv"); assert list(n4["clip"]) == list(pm["clip"]); NNk = np.load(f"{PC}/pg_private_NNk.npy")
P0 = (7 * np.load(f"{PC}/pg_private_b2G.npy") + NNk) / 8; assert np.allclose(P0, np.load(f"{PC}/pg_private_b2GNNk.npy")) and (cg[P0.argmax(1)] == n4["goalkeeper_direction"].values).all(), "N4 keeper check"
Pg = (7 * np.load(f"{PC}/nov6/pg_private_b2GD.npy") + NNk) / 8 if "g3" in FL else P0
sub = n4.assign(goalkeeper_direction=cg[Pg.argmax(1)]); sub.to_csv(f"{P}/submission_{NAME}.csv", index=False); np.save(f"{PC}/pg_private_{NAME}.npy", Pg)
if not FL: assert open(f"{P}/submission_{NAME}.csv", "rb").read() == open(f"{P}/submission_N4.csv", "rb").read(), "no flags must rebuild submission_N4.csv"
chg = sub.goalkeeper_direction.values != n4.goalkeeper_direction.values
print(f"wrote submission_{NAME}.csv | {'new keeper (G3)' if FL else 'N4 as is'} | kick changed vs N4 0 | keeper changed on {int(chg.sum())} of {len(sub)} "
      f"({pd.Series([a[:5] + '->' + b[:5] for a, b in zip(n4.goalkeeper_direction.values[chg], sub.goalkeeper_direction.values[chg])]).value_counts().to_dict()}) | keeper counts {sub.goalkeeper_direction.value_counts().to_dict()}")
