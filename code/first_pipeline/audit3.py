import pandas as pd, numpy as np
C = r"C:\path\to\project-root\Q4_work\cache"; cov = pd.read_pickle(f"{C}/audit2.pkl")
cov["size_bin"] = pd.qcut(cov["kh"], 3, labels=["small (<81px)", "medium (81-98px)", "large (>98px)"])
print("kick acc / keeper acc / n  by kicker-size band x healthy")
t = cov.groupby(["size_bin", "healthy"], observed=True).agg(n=("okk", "size"), kick=("okk", "mean"), keeper=("okg", "mean")).round(3); print(t.to_string())
print("\nkick acc by size band only:", cov.groupby("size_bin", observed=True)["okk"].mean().round(3).to_dict())
print("share of unhealthy clips by size band:", (1 - cov.groupby("size_bin", observed=True)["healthy"].mean()).round(2).to_dict())
# counterfactual: unhealthy clips performing like healthy clips of the same size band
h = cov[cov.healthy].groupby("size_bin", observed=True)[["okk", "okg"]].mean(); cf_k = cov["size_bin"].map(h["okk"]).astype(float).where(~cov.healthy, cov.okk.astype(float)); cf_g = cov["size_bin"].map(h["okg"]).astype(float).where(~cov.healthy, cov.okg.astype(float))
base = 0.6 * cov.okk.mean() + 0.25 * cov.okg.mean() + 0.15 * cov.goal.mean(); cf = 0.6 * cf_k.mean() + 0.25 * cf_g.mean() + 0.15 * cov.goal.mean()
print(f"\ncounterfactual (unhealthy clips as good as healthy clips of the same size): kick {cov.okk.mean():.3f}->{cf_k.mean():.3f} keeper {cov.okg.mean():.3f}->{cf_g.mean():.3f} score {base:.3f}->{cf:.3f} (+{cf - base:.3f})")
print(f"upper bound if every clip performed like the large-player clips: kick {cov[cov.size_bin == 'large (>98px)'].okk.mean():.3f} (now {cov.okk.mean():.3f}) -> +{0.6 * (cov[cov.size_bin == 'large (>98px)'].okk.mean() - cov.okk.mean()):.3f} score")
print("\nkeeper-leg failures: keeper height in px for weak vs ok:", cov.groupby(cov.keep_legs_ok < .9)["gh"].median().round(0).to_dict(), "| share weak when keeper < 70px:", round((cov[cov.gh < 70].keep_legs_ok < .9).mean(), 2), "| when keeper > 100px:", round((cov[cov.gh > 100].keep_legs_ok < .9).mean(), 2))
print("ball support<3 vs kicker height:", cov.groupby(cov.ball_support < 3)["kh"].median().round(0).to_dict())
