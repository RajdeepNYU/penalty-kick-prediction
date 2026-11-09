"""P4rkT + mean of camera-aware pixel models (pxlib.py), each fitted on all public clips. Kick = (1 - w) * P4rkT + w * mean(pixel); keeper = P4rkT's. Private clips without features keep P4rkT. Each model's private probabilities are cached, so a new blend does not refit. Writes only the local file; submitting is a separate step.
usage: python make_subPx2.py NAME w views:k:C [views:k:C ...] (views joined by commas) -> submission_{NAME}.csv, cache/Pk_private_{NAME}.npy, cache/Pp_{views}_{k}_{C}.npy"""
import sys, os, numpy as np, pandas as pd
from common import *
import pxlib
from pxlib import cl, prv

NAME, w = sys.argv[1], float(sys.argv[2]); Pb = np.load(f"{PC}/Pk_private_P4rkT.npy"); base = pd.read_csv(f"{P}/submission_P4rkT.csv"); assert list(base["clip"]) == prv and (cl[Pb.argmax(1)] == base["kick_direction"].values).all(); Ps = []; OK = np.ones(len(prv), bool)
for sp in sys.argv[3:]:
    vs, k, C = sp.split(":"); fp = f"{PC}/Pp_{vs.replace(',', '+')}_{k}_{C}.npy"
    if os.path.exists(fp): z = np.load(fp, allow_pickle=True).item(); Px, oq = z["P"], z["ok"]
    else: Px, oq = pxlib.private(vs.split(","), int(k), float(C)); np.save(fp, dict(P=Px, ok=oq))
    Ps.append(Px); OK &= oq; print(f"{sp}: pixel alone {pd.Series(cl[Px.argmax(1)]).value_counts().to_dict()} | agreement with P4rkT {np.mean(cl[Px.argmax(1)] == base.kick_direction.values):.3f} | no features {int((~oq).sum())}", flush=True)
F = sum(Ps) / len(Ps); Pk = np.where(OK[:, None], (1 - w) * Pb + w * F, Pb); np.save(f"{PC}/Pk_private_{NAME}.npy", Pk); sub = base.assign(kick_direction=cl[Pk.argmax(1)]); sub.to_csv(f"{P}/submission_{NAME}.csv", index=False); chg = sub.kick_direction != base.kick_direction
print(f"wrote submission_{NAME}.csv | kick changed on {int(chg.sum())} of {len(sub)} ({base.kick_direction[chg].str[:1].add('->').add(sub.kick_direction[chg].str[:1]).value_counts().to_dict()}) | counts {sub.kick_direction.value_counts().to_dict()}")
