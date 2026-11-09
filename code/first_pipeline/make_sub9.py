"""v9: kick = 0.5 * mean(trees on RTMW features, trees on ViTPose-large features) + 0.5 * temporal-network bag trained on four pose-measurement streams (RTMW, RTMW3D-2D, ViTPose base, ViTPose large). Keeper/outcome unchanged from v7."""
import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
pm = pd.read_csv(f"{D}/private_metadata.csv"); tmpl = pd.read_csv(D + r"\submission_template.csv")
trees = 0.5 * np.load(f"{C}/pk_private_v7.npy") + 0.5 * np.load(f"{C}/trees_private_F.npy"); Pk = 0.5 * trees + 0.5 * np.load(f"{C}/nn_private_EF.npy"); Pg = np.load(f"{C}/pg_private_v7.npy")
ck = np.array(["left", "middle", "right"]); cg = np.array(["left", "right", "stands in the middle"])
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": ck[Pk.argmax(1)], "goalkeeper_direction": cg[Pg.argmax(1)], "goal_outcome": "goal"})
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique and set(sub["kick_direction"]) <= set(ck) and set(sub["goalkeeper_direction"]) <= set(cg)
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(ROOT + r"\Q4_work\submission_v9.csv", index=False); v7 = pd.read_csv(ROOT + r"\Q4_work\submission_v7.csv")
print("wrote submission_v9.csv | agreement with v7: kick %.3f keeper %.3f" % ((v7.kick_direction == sub.kick_direction).mean(), (v7.goalkeeper_direction == sub.goalkeeper_direction).mean()))
