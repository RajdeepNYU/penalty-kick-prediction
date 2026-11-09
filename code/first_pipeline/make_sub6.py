"""v6 submission = v5 with a stronger kick model.
kick    = 0.5 * mean{GBDT bag, ExtraTrees on 259 kinematic features}  +  0.5 * temporal pose-network bag (5 configs x 3 seeds, two pose streams, mirror aug/TTA)
keeper  = unchanged from v5 (mean of GBDT, ExtraTrees on camera-stabilised keeper features, keeper CNN)
outcome = always 'goal'."""
import numpy as np, pandas as pd
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
pm = pd.read_csv(f"{D}/private_metadata.csv"); tmpl = pd.read_csv(D + r"\submission_template.csv")
Pk_trees = np.load(f"{C}/pk_private_v5.npy"); Pk_nn = np.load(f"{C}/nn_private.npy"); Pg = np.load(f"{C}/pg_private_v5.npy")      # trees: (left, middle, right) ; keeper: (left, right, stands)
Pk = 0.5 * Pk_trees + 0.5 * Pk_nn; ck = np.array(["left", "middle", "right"]); cg = np.array(["left", "right", "stands in the middle"])
sub = pd.DataFrame({"clip": pm["clip"], "kick_direction": ck[Pk.argmax(1)], "goalkeeper_direction": cg[Pg.argmax(1)], "goal_outcome": "goal"})
assert list(sub["clip"]) == list(tmpl["clip"]) and len(sub) == 900 and sub["clip"].is_unique and set(sub["kick_direction"]) <= set(ck) and set(sub["goalkeeper_direction"]) <= set(cg)
for k in ("kick_direction", "goalkeeper_direction", "goal_outcome"): print(k, {str(a): int(b) for a, b in sub[k].value_counts().items()})
sub.to_csv(ROOT + r"\Q4_work\submission_v6.csv", index=False); v5 = pd.read_csv(ROOT + r"\Q4_work\submission_v5.csv")
print("wrote submission_v6.csv | agreement with v5: kick %.3f keeper %.3f" % ((v5.kick_direction == sub.kick_direction).mean(), (v5.goalkeeper_direction == sub.goalkeeper_direction).mean()))
