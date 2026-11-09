"""v10: kick = 0.5 * mean(trees on RTMW / ViTPose-large / ViTPose-huge kinematic features) + 0.5 * mean(temporal-network bags: streams RTMW+RTMW3D+ViTPose-base+large, and RTMW+RTMW3D+ViTPose-huge).
keeper = (RTMW keeper trees + ViTPose-large keeper trees + keeper CNN) weighted 2:2:1; outcome = always goal."""
import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
pm = pd.read_csv(f"{D}/private_metadata.csv"); tmpl = pd.read_csv(D + r"\submission_template.csv")
trees = (np.load(f"{C}/pk_private_v7.npy") + np.load(f"{C}/trees_private_F.npy") + np.load(f"{C}/trees_private_G.npy")) / 3; nn = (np.load(f"{C}/nn_private_EF.npy") + np.load(f"{C}/nn_private_G.npy")) / 2; Pk = 0.5 * trees + 0.5 * nn
Pg = (3 * np.load(f"{C}/pg_private_v7.npy") + np.load(f"{C}/keeper_private_F.npy")) / 5      # v7 keeper = (HGB+ET+CNN)/3 -> (HGB+ET+HGB_F+ET_F+CNN)/5
ck = np.array(["left", "middle", "right"]); cg = np.array(["left", "right", "stands in the middle"])
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": ck[Pk.argmax(1)], "goalkeeper_direction": cg[Pg.argmax(1)], "goal_outcome": "goal"})
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique and set(sub["kick_direction"]) <= set(ck) and set(sub["goalkeeper_direction"]) <= set(cg)
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(ROOT + r"\Q4_work\submission_v10.csv", index=False); v9 = pd.read_csv(ROOT + r"\Q4_work\submission_v9.csv")
print("wrote submission_v10.csv | agreement with v9: kick %.3f keeper %.3f" % ((v9.kick_direction == sub.kick_direction).mean(), (v9.goalkeeper_direction == sub.goalkeeper_direction).mean()))
