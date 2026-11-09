"""Test the temporal-median ball method on the clips where the consensus ball has <3 supporting frames, and compare nearest-foot consistency."""
import pickle, numpy as np, pandas as pd, sys
sys.argv = ["x"]
R = r"C:\path\to\project-root"; C = R + r"\Q4_work\cache"
cov = pd.read_pickle(f"{C}/audit_cov.pkl"); weak = cov[cov.ok & (cov.ball_support < 3)]["clip"].tolist(); print("weak-ball clips (support<3):", len(weak))
src = open(f"{R}\\Q4_work\\ballmed.py", encoding="utf-8").read().replace("if len(sys.argv) > 1: todo = todo[::max(len(todo) // int(sys.argv[1]), 1)][:int(sys.argv[1])]", "todo = [t for t in todo if t[1] in WEAK]").replace('OUT = f"{C}/ballmed"', 'OUT = f"{C}/ballmed_weak"')
exec(compile(src, "ballmed_weak", "exec"), {"__name__": "__main__", "WEAK": set(weak), "sys": sys})
# evaluate: distance between ball estimate and the nearest foot keypoint in the last frame (self-consistency), median method vs consensus
rows = []
for c in weak:
    m = pickle.load(open(f"{C}/ballmed_weak/{c}.pkl", "rb")); b = pickle.load(open(f"{C}/ballspot/{c}.pkl", "rb")); k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb"))
    if m is None or b is None: continue
    M = b["M"][-1]; P = k["kp"][-1][[15, 16, 17, 19, 20, 22]]; hL = m["hL"]
    d = lambda ball: np.hypot(*(P - ball).T).min() / hL if np.isfinite(ball[0]) else np.nan
    rows.append(dict(found_med=bool(np.isfinite(m["ball_med"][0])), conf=m["conf"], d_cons=d(b["ball_last"]), d_med=d(m["ball_med"]), agree=m["agree"]))
r = pd.DataFrame(rows)
print(f"median method finds a ball in {r.found_med.mean():.1%} of these clips; near a foot (<0.5 body-heights): consensus {(r.d_cons < .5).mean():.1%} vs median {(r.d_med < .5).mean():.1%}; median method conf>=0.1: {(r.conf >= .1).mean():.1%}")
print("median distance to nearest foot (body heights): consensus %.2f, median-method %.2f" % (r.d_cons.median(), r.d_med.median()))
