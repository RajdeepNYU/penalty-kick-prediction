import pickle, numpy as np, pandas as pd
R = r"C:\path\to\project-root"; D = R + r"\Q4data"; C = R + r"\Q4_work\cache"
cov = pd.read_pickle(f"{C}/audit_cov.pkl"); a = cov[cov.ok].reset_index(drop=True); LEG = [11, 12, 13, 14, 15, 16]; rows = []
for c in a["clip"]:
    k = pickle.load(open(f"{C}/kpts/{c}.pkl", "rb")); sc = k["sc"][-8:][:, LEG]; med = np.median(sc, axis=0); rows.append(dict(worst_joint_med=med.min(), worst_joint=LEG[int(med.argmin())], frac_below=(sc < 2.0).mean(), hL=k["box"][-1][3] - k["box"][-1][1]))
a = pd.concat([a, pd.DataFrame(rows)], axis=1); weak = a[a.kick_legs_ok < 0.9]
print(f"clips with weak kicker legs: {len(weak)} of {len(a)} ({len(weak) / len(a):.1%})")
print("their worst-joint median score percentiles 10/50/90:", np.percentile(weak.worst_joint_med, [10, 50, 90]).round(2), "| all clips:", np.percentile(a.worst_joint_med, [10, 50, 90]).round(2))
for thr in (2.0, 1.8, 1.6, 1.4, 1.2): print(f"  if the leg-confidence cutoff were {thr}: weak clips = {(a.worst_joint_med < thr).mean():.1%}")
print("worst joint index among weak clips (11/12 hips, 13/14 knees, 15/16 ankles):", weak.worst_joint.value_counts().to_dict())
print("kicker height (px) weak vs ok:", weak.hL.median().round(0), a[a.kick_legs_ok >= .9].hL.median().round(0))
