"""Synthetic camera views as extra training rows: the learning curve gains about 2 points per doubling of real clips and only mirroring multiplies them now. With Q4_VIEW=j in the environment, common.load("ballspot"), common.tracks() and streams.stream_kb
return every clip through a small per-clip image-plane map (seeded by clip name and j): roll ~ N(0, ROLL deg), vertical scale ~ exp N(0, SY) (camera elevation), shear ~ N(0, SH). Stabilised keypoints follow through M, so the feature code runs unchanged.
build: cache_vJ = ball2 and camera2 moved through the same map, then kin2b / kinR A, F, G.   eval: trees (HGB + ET, folds as cvlib): aug = train on base + V1 + V2, test on base; tta = same training, test on the mean over base, V1, V2.
usage: python viewaug.py build | eval [3,4,5] -> logs/viewaug.txt"""
import os, sys, pickle, zlib, numpy as np

ROLL, SY, SH, NV = 3.0, 0.12, 0.08, 2
J = int(os.environ.get("Q4_VIEW", "0"))


def lin(c, j=None):
    r = np.random.default_rng(zlib.crc32(c.encode()) + 1000 * (J if j is None else j)); a = np.radians(r.normal(0, ROLL)); sy = float(np.exp(r.normal(0, SY))); k = r.normal(0, SH)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1.0, k], [0.0, sy]]), sy


def view_b(c, b):
    A, _ = lin(c); H = np.eye(3); H[:2, :2] = A
    return dict(b, M=(H @ np.asarray(b["M"], float)).astype(np.float32), ball_last=A @ np.asarray(b["ball_last"], float))


def view_kb(c, kb):
    _, sy = lin(c); bx = np.array(kb["box"], float); bx[..., [1, 3]] *= sy
    return dict(kb, box=bx)


def view_tr(c, t):
    if t is None or t.get("keep") is None: return t
    A, sy = lin(c); kb = np.array(t["keep"], float); cx, y2 = (kb[:, 0] + kb[:, 2]) / 2, kb[:, 3]; w, h = kb[:, 2] - kb[:, 0], (kb[:, 3] - kb[:, 1]) * sy; p = np.stack([cx, y2], 1) @ A.T
    return dict(t, keep=np.stack([p[:, 0] - w / 2, p[:, 1] - h, p[:, 0] + w / 2, p[:, 1]], 1))


if __name__ == "__main__":
    from common import *
    tag = lambda j: f"{P}/cache_v{j}"
    if sys.argv[1] == "build":
        from runner import sh as _sh
        LOG = open(f"{P}/logs/viewaug_build.log", "a", buffering=1); B2 = pickle.load(open(f"{PC}/ball2.pkl", "rb")); CAM = pickle.load(open(f"{PC}/camera2.pkl", "rb"))
        for j in range(1, NV + 1):
            os.makedirs(tag(j), exist_ok=True); B, C_ = {}, {}
            for c in CLIPS:
                A, _ = lin(c, j); r = B2.get(c); B[c] = None if r is None else dict(r, ball=A @ np.asarray(r["ball"], float)); q = CAM.get(c)
                if q is not None: Ac = A @ np.asarray(q["A"], float); C_[c] = dict(q, A=Ac, Ai=np.linalg.inv(Ac), az=A @ np.asarray(q["az"], float), ball_ground=A @ np.asarray(q["ball_ground"], float))
            pickle.dump(B, open(f"{tag(j)}/ball2.pkl", "wb")); pickle.dump(C_, open(f"{tag(j)}/camera2.pkl", "wb")); E = dict(Q4P_CACHE=tag(j), Q4_VIEW=str(j))
            for S in "AFG": _sh(LOG, ["kin2b.py", S, "ball2"], E); _sh(LOG, ["kinR.py", S], E)
    if sys.argv[1] == "eval":
        from cvlib import cl, yk, ym, grp, hg, et, SW, stack
        from sklearn.model_selection import StratifiedGroupKFold
        pub = list(gt["clip"]); sd = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 1, 2); SF = "" if sd == (0, 1, 2) else "_sd345"

        def feats(root, S):
            A, B = [], []
            for f in (f"kin2_{S}_ball2", f"kinR_{S}"):
                K = pickle.load(open(f"{root}/{f}.pkl", "rb")); r = K["rows"]
                if isinstance(next(iter(r.values()))[0], tuple): r = {c: (v[0][0], v[1][0]) for c, v in r.items()}
                a, b = stack(r, pub, len(K["names"])); A.append(a); B.append(b)
            return np.concatenate(A, 1), np.concatenate(B, 1)

        ok = lambda Q: cl[Q.argmax(1)] == yk; R = {k: 0 for k in ("base", "aug", "tta")}
        for S in "AFG":
            Xs = [feats(PC if j == 0 else tag(j), S) for j in range(NV + 1)]; keep = ~np.isnan(Xs[0][0]).all(0); Xs = [(a[:, keep], b[:, keep]) for a, b in Xs]; Pa, Pt = np.zeros((len(pub), 3)), np.zeros((len(pub), 3))
            for mk in (hg, et):
                for s in sd:
                    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(Xs[0][0], yk, grp):
                        m = mk(s).fit(np.concatenate([q[tr] for x in Xs for q in x]), np.concatenate([np.r_[yk[tr], ym[tr]]] * len(Xs))); c_ = list(m.classes_); ix = [c_.index(SW[k]) for k in c_]; o = [c_.index(k) for k in cl]
                        pr = lambda Qs: sum(((m.predict_proba(a[te]) + m.predict_proba(b[te])[:, ix]) / 2)[:, o] for a, b in Qs) / len(Qs); Pa[te] += pr(Xs[:1]) / (2 * len(sd)); Pt[te] += pr(Xs) / (2 * len(sd))
            P0 = (np.load(f"{PC}/oof_hgb_{S}_b2R{SF}.npy") + np.load(f"{PC}/oof_et_{S}_b2R{SF}.npy")) / 2; print(f"{S}: base {ok(P0).mean():.4f} | aug {ok(Pa).mean():.4f} | aug + tta {ok(Pt).mean():.4f}", flush=True)
            for k, v in (("base", P0), ("aug", Pa), ("tta", Pt)): R[k] = R[k] + v / 3
            np.save(f"{PC}/oof_view_aug_{S}{SF}.npy", Pa); np.save(f"{PC}/oof_view_tta_{S}{SF}.npy", Pt)
        b = ok(R["base"]); print(f"\nblend of A, F, G, seeds {sd}")
        for k in ("base", "aug", "tta"):
            q = ok(R[k]); print(f"  {k:6s} kick {q.mean():.4f} L/R {q[yk != 'middle'].mean():.4f} | fixed {int((q & ~b).sum())} broken {int((b & ~q).sum())} net {int(q.sum() - b.sum()):+d}")
