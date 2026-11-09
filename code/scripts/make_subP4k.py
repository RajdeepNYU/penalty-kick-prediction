"""P4 + keeper NN: kick column of submission_P4.csv unchanged; keeper = (7 * pg_private_b2G + pg_private_NNk) / 8, i.e. the P4 keeper's 7 members plus the
keeper temporal NN (pipeline23.py, 10 full-mode runs) as an eighth member (CV: keeper 0.9215 -> 0.9238, +3/-0 clips). -> submission_P4k.csv (not submitted)"""
import numpy as np, pandas as pd
from common import *

cg = np.array(["left", "right", "stands in the middle"])
base = pd.read_csv(f"{P}/submission_P4.csv"); Pb = np.load(f"{PC}/pg_private_b2G.npy"); Pg = (7 * Pb + np.load(f"{PC}/pg_private_NNk.npy")) / 8
assert list(base["clip"]) == list(pm["clip"]) and (cg[Pb.argmax(1)] == base["goalkeeper_direction"].values).all()      # P4's keeper column is pg_private_b2G
sub = base.assign(goalkeeper_direction=cg[Pg.argmax(1)]); sub.to_csv(f"{P}/submission_P4k.csv", index=False); np.save(f"{PC}/pg_private_b2GNNk.npy", Pg)
ch = sub.goalkeeper_direction != base.goalkeeper_direction
print(f"wrote submission_P4k.csv | keeper {sub.goalkeeper_direction.value_counts().to_dict()} | keeper changed on {int(ch.sum())} private clips:",
      base.goalkeeper_direction[ch].str[:5].add(" -> ").add(sub.goalkeeper_direction[ch].str[:5]).value_counts().to_dict())
