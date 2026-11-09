"""Stage D3b (any number of pose-measurement streams via env STREAMS, e.g. EF): final temporal pose network bag, trained on ALL public clips (RTMW stream + RTMW3D-2D stream as two training views, original + mirrored, label-swapped),
predicting the private clips (both streams x mirror, averaged). Saves cache/nn_private.npy [900,3] in order (left, middle, right)."""
import os, pickle, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
CONFIGS = [(12, 96), (15, 128), (20, 96), (20, 128), (30, 128)]; SEEDS = (0, 1, 2)
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); pm = pd.read_csv(f"{D}/private_metadata.csv")
STREAMS = os.environ.get("STREAMS", ""); OUTF = os.environ.get("NN_OUT", "nn_private2.npy")
RA, RB = pickle.load(open(f"{C}/seq.pkl", "rb")), pickle.load(open(f"{C}/seqB.pkl", "rb")); RX = [pickle.load(open(f"{C}/seq{L}.pkl", "rb")) for L in STREAMS]
def stack(R, clips): return [np.stack([R[c][j // 2][j % 2] for c in clips]) for j in range(4)]       # X, S, Xm, Sm
TR = [stack(R, gt["clip"]) for R in [RA, RB] + RX]; PR = [stack(R, pm["clip"]) for R in [RA, RB] + RX]; A = TR[0]
y = gt["kick_direction"].map({"left": 0, "middle": 1, "right": 2}).values; ym = np.where(y == 1, 1, 2 - y)
dev = "cuda"; Nc = A[0].shape[2]; ST = A[1].shape[1]; NS_ = len(TR)


class Net(nn.Module):
    def __init__(s, H):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(Nc, H), nn.GELU(), nn.Dropout(0.1))
        s.convs = nn.ModuleList([nn.Conv1d(H, H, 3, padding=d, dilation=d) for d in (1, 2, 1)]); s.drop = nn.Dropout(0.2); s.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        s.head = nn.Sequential(nn.Linear(3 * H + ST, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(s, x, st):
        h = s.inp(x)
        for conv, nm in zip(s.convs, s.norm): h = nm(h + s.drop(F.gelu(conv(h.transpose(1, 2)).transpose(1, 2))))
        return s.head(torch.cat([h[:, -1], h[:, -6:].mean(1), h.max(1).values, st], 1))


def fit_predict(EP, H, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    both = np.concatenate([t[j] for t in TR for j in (0, 2)]); mu = both.reshape(-1, Nc).mean(0); sd = both.reshape(-1, Nc).std(0) + 1e-3; sd[-3:] = 1; mu[-3:] = 0
    nz = lambda a: np.clip((a - mu) / sd, -6, 6).astype(np.float32); T = lambda a: torch.tensor(a, device=dev)
    tX, tXm = T(np.concatenate([nz(t[0]) for t in TR])), T(np.concatenate([nz(t[2]) for t in TR])); tS, tSm = T(np.concatenate([t[1] for t in TR])), T(np.concatenate([t[3] for t in TR]))
    ty, tym = T(np.concatenate([y] * NS_)), T(np.concatenate([ym] * NS_)); n = len(tX)
    m = Net(H).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EP * ((n + 63) // 64), pct_start=0.15)
    for ep in range(EP):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            ix = perm[i:i + 64]; fl = torch.rand(len(ix), device=dev) < 0.5
            xb = torch.where(fl[:, None, None], tXm[ix], tX[ix]); sb = torch.where(fl[:, None], tSm[ix], tS[ix]); yb = torch.where(fl, tym[ix], ty[ix])
            xb = xb + 0.03 * torch.randn_like(xb) * (xb != 0)
            loss = F.cross_entropy(m(xb, sb), yb, label_smoothing=0.1); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval(); outs = []
    with torch.no_grad():
        for P_ in PR:
            e = [T(nz(P_[0])), T(nz(P_[2])), T(P_[1]), T(P_[3])]
            outs.append((F.softmax(m(e[0], e[2]), 1) + F.softmax(m(e[1], e[3]), 1)[:, [2, 1, 0]]) / 2)
    return (sum(outs) / len(outs)).cpu().numpy()


t0 = time.time(); tot = 0
for EP, H in CONFIGS:
    for s in SEEDS: tot = tot + fit_predict(EP, H, s)
    print(f"config epochs={EP} hidden={H} done [{time.time() - t0:.0f}s]", flush=True)
P = tot / (len(CONFIGS) * len(SEEDS)); np.save(f"{C}/{OUTF}", P); print("saved", P.shape, "pred counts", {k: int(v) for k, v in zip(["left", "middle", "right"], np.bincount(P.argmax(1), minlength=3))})
