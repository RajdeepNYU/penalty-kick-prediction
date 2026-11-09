"""Trees CV on one or more feature caches concatenated column-wise (same protocol as Q4_work). usage: python e_trees.py TAG feat1 [feat2 ...]
feat = cache name in Q4_word_Prime/cache (without .pkl), or W:name for a Q4_work cache. All-NaN train columns are dropped.
Saves cache/oof_hgb_TAG.npy, cache/oof_et_TAG.npy and prints accuracy vs the matching v10 base trees when given --ref SUFFIX."""
import sys, pickle, numpy as np
from common import *
from cvlib import *

args = [a for a in sys.argv[1:] if not a.startswith("--")]; opts = dict(a[2:].split("=") for a in sys.argv[1:] if a.startswith("--"))
TAG, feats = args[0], args[1:]
Xa, Xb = [], []
for f in feats:
    p = f"{W}/{f[2:]}.pkl" if f.startswith("W:") else f"{PC}/{f}.pkl"; K = pickle.load(open(p, "rb")); r = K["rows"]; dim = len(K["names"])
    if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}          # kinR rows: (static, seq, nst)
    a, b = stack(r, gt["clip"], dim); Xa.append(a); Xb.append(b)
Xa, Xb = np.concatenate(Xa, 1), np.concatenate(Xb, 1); keep = ~np.isnan(Xa).all(0); Xa, Xb = Xa[:, keep], Xb[:, keep]
aug = []
for grp_ in (opts["aug"].split(",") if "aug" in opts else []):                 # time-shifted windows of the same public clips; "+" joins caches in the order of feats
    A_, B_, ok = [], [], np.ones(len(gt), bool)
    for f in grp_.split("+"):
        K = pickle.load(open(f"{PC}/{f}.pkl", "rb")); r = K["rows"]
        if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}
        a, b = stack(r, gt["clip"], len(K["names"])); A_.append(a); B_.append(b); ok &= np.array([c in r for c in gt["clip"]])
    aug.append((np.concatenate(A_, 1)[:, keep], np.concatenate(B_, 1)[:, keep], ok))
print(TAG, "features", Xa.shape[1], "| aug sets", len(aug))
sd = tuple(int(s) for s in opts.get("seeds", "0,1,2").split(","))              # --seeds=3,4,5: other partitions / model seeds (noise check)
H = oof_aug(hg, Xa, Xb, aug, seeds=sd) if aug else oof(hg, Xa, Xb, seeds=sd); E = oof_aug(et, Xa, Xb, aug, seeds=sd) if aug else oof(et, Xa, Xb, seeds=sd); np.save(f"{PC}/oof_hgb_{TAG}.npy", H); np.save(f"{PC}/oof_et_{TAG}.npy", E); T_ = (H + E) / 2
print(f"  {TAG}: HGB {acc(H):.4f} ET {acc(E):.4f} trees {acc(T_):.4f} (L/R {lr_acc(T_):.4f})")
if opts.get("lgb"):
    L = oof_aug(lgb, Xa, Xb, aug) if aug else oof(lgb, Xa, Xb); np.save(f"{PC}/oof_lgb_{TAG}.npy", L)
    print(f"  {TAG}: LGB {acc(L):.4f} | HGB+ET+LGB {acc((H + E + L) / 3):.4f} | ET+LGB {acc((E + L) / 2):.4f}")
if "ref" in opts:
    s = opts["ref"] if opts["ref"] != "A" else ""; Hr, Er = np.load(f"{W}/base_hgb{s}.npy"), np.load(f"{W}/base_et{s}.npy"); R = (Hr + Er) / 2
    print(f"  ref v10 base{s}: HGB {acc(Hr):.4f} ET {acc(Er):.4f} trees {acc(R):.4f} (L/R {lr_acc(R):.4f}) | delta trees {acc(T_) - acc(R):+.4f}")
