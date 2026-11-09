"""Derives train_seqk.py (keeper temporal network CV) from train_seq.py."""
p = r"C:\path\to\project-root\Q4_work"; s = open(f"{p}\\train_seq.py", encoding="utf-8").read()
def rep(a, b):
    global s; assert a in s, a[:60]; s = s.replace(a, b)
rep('rows = pickle.load(open(f"{C}/seq.pkl", "rb"))', 'rows = pickle.load(open(f"{C}/seqk.pkl", "rb"))\nZ = (np.zeros((12, 49), np.float32), np.zeros(4, np.float32))\nfor c in gt["clip"]: rows.setdefault(c, (Z, Z))')
rep('y = gt["kick_direction"].map(lab).values', 'y = gt["goalkeeper_direction"].map(lab).values')
rep('lab = {"left": 0, "middle": 1, "right": 2}', 'lab = {"left": 0, "stands in the middle": 1, "right": 2}')
rep('gt["kick_direction"].values, grp)', 'gt["goalkeeper_direction"].values, grp)')
rep('RB = pickle.load(open(f"{C}/seqB.pkl", "rb"))', 'RB = pickle.load(open(f"{C}/seqkB.pkl", "rb"))\n    for c in gt["clip"]: RB.setdefault(c, (Z, Z))')
rep('oof_seq_', 'oof_seqk_')
i = s.index('cl = np.array(["left", "middle", "right"]); pred')
s = s[:i] + '''cl = np.array(["left", "stands in the middle", "right"]); pred = cl[P.argmax(1)]; yk = gt["goalkeeper_direction"].values
print(f"KEEPER SEQ NET (epochs {EP}, hidden {H}, augB {AUGB}): keeper acc {(pred == yk).mean():.3f}")
np.save(f"{C}/oof_seqk_{EP}_{H}_b{AUGB}.npy", P)
M = pickle.load(open(f"{C}/oof_keep_lab.pkl", "rb")); acc = lambda Q: (cl[Q.argmax(1)] == yk).mean()
trees = (M["new_hgb"] + M["new_et"])[:, [0, 2, 1]] / 2; cnn = M["cnn"][:, [0, 2, 1]]; cur = (trees * 2 + cnn) / 3
print(f"current keeper blend (2 trees + CNN) {acc(cur):.3f} | + seq NN (equal 1/4) {acc(cur * 0.75 + P * 0.25):.3f} | 1/3 {acc(cur * 0.667 + P * 0.333):.3f} | 1/2 {acc(cur * 0.5 + P * 0.5):.3f}")
'''
open(f"{p}\\train_seqk.py", "w", encoding="utf-8").write(s); print("wrote train_seqk.py")
