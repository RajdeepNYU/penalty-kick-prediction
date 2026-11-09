"""Stage D2: temporal pose network for kick direction. Source-grouped 5-fold CV (3 partitions x 1 model each), mirror aug (label swap) + mirror TTA.
usage: python train_seq.py [epochs] [hidden]   -> writes cache/oof_seq.npy (OOF probs left,middle,right) and prints accuracies"""
import pickle, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
import os; EP = int(sys.argv[1]) if len(sys.argv) > 1 else 60; H = int(sys.argv[2]) if len(sys.argv) > 2 else 96; KIN = int(os.environ.get("KIN", 0)); SPF = int(os.environ.get("SPF", 1)); ARCH = os.environ.get("ARCH", "conv")
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); rows = pickle.load(open(f"{C}/seq.pkl", "rb"))
X = np.stack([rows[c][0][0] for c in gt["clip"]]); S = np.stack([rows[c][0][1] for c in gt["clip"]]); Xm = np.stack([rows[c][1][0] for c in gt["clip"]]); Sm = np.stack([rows[c][1][1] for c in gt["clip"]])
lab = {"left": 0, "middle": 1, "right": 2}; y = gt["kick_direction"].map(lab).values; grp = gt["source_id"].values; ym = np.where(y == 1, 1, 2 - y)
AUGB = int(os.environ.get("AUGB", 0))
if AUGB:                                                      # second measurement stream (RTMW3D's 2D keypoints) as extra training samples + averaged test predictions
    RB = pickle.load(open(f"{C}/seqB.pkl", "rb"))
    XB = np.stack([RB[c][0][0] for c in gt["clip"]]); SB = np.stack([RB[c][0][1] for c in gt["clip"]]); XBm = np.stack([RB[c][1][0] for c in gt["clip"]]); SBm = np.stack([RB[c][1][1] for c in gt["clip"]])
if KIN:                                                       # also feed the 259 hand-built kinematic features (kin.pkl) to the head
    KR = pickle.load(open(f"{C}/kin.pkl", "rb"))["rows"]; nanv = np.full(259, np.nan, np.float32)
    S = np.c_[S, np.stack([KR.get(c, (nanv, nanv))[0] for c in gt["clip"]])]; Sm = np.c_[Sm, np.stack([KR.get(c, (nanv, nanv))[1] for c in gt["clip"]])]
EXTRA = []; STREAMS = os.environ.get("STREAMS", "")          # additional measurement streams (seqC.pkl, seqD.pkl, ...) used like stream B
for L_ in STREAMS:
    RL = pickle.load(open(f"{C}/seq{L_}.pkl", "rb")); EXTRA.append(tuple(np.stack([RL[c][j // 2][j % 2] for c in gt["clip"]]) for j in (0, 2, 1, 3)))
KEEPCH = int(os.environ.get("KEEPCH", 0)); AUX = int(os.environ.get("AUX", 0)); JIT = int(os.environ.get("JIT", 0))
Zk = (np.zeros((12, 49), np.float32), np.zeros(4, np.float32))
def _kd(R): return np.stack([R.get(c, (Zk, Zk))[0][0] for c in gt["clip"]]), np.stack([R.get(c, (Zk, Zk))[1][0] for c in gt["clip"]])
if KEEPCH:                                                    # append the keeper's per-frame channels to the kicker's
    RK = pickle.load(open(f"{C}/seqk.pkl", "rb")); XK, XKm = _kd(RK); X = np.concatenate([X, XK], 2); Xm = np.concatenate([Xm, XKm], 2)
    if AUGB:
        RKB = pickle.load(open(f"{C}/seqkB.pkl", "rb")) if os.path.exists(f"{C}/seqkB.pkl") else RK; XKB, XKBm = _kd(RKB); XB = np.concatenate([XB, XKB], 2); XBm = np.concatenate([XBm, XKBm], 2)
labg = {"left": 0, "stands in the middle": 1, "right": 2}; yg = gt["goalkeeper_direction"].map(labg).values; ygm = np.where(yg == 1, 1, 2 - yg)
dev = "cuda"; Nc = X.shape[2]; ST = S.shape[1]


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(Nc, H), nn.GELU(), nn.Dropout(0.1))
        s.convs = nn.ModuleList([nn.Conv1d(H, H, 3, padding=d, dilation=d) for d in (1, 2, 1)]); s.drop = nn.Dropout(0.2); s.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(3)])
        if ARCH == "gru": s.rnn = nn.GRU(H, H, batch_first=True, bidirectional=True); s.proj = nn.Linear(2 * H, H)
        if ARCH == "tf": s.pos = nn.Parameter(torch.zeros(X.shape[1], H)); s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(H, 4, 2 * H, 0.2, batch_first=True, norm_first=True), 2)
        s.head = nn.Sequential(nn.Linear(3 * H + ST, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3)); s.aux = nn.Sequential(nn.Linear(3 * H + ST, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(s, x, st, aux=False):                        # x [B,T,C]
        h = s.inp(x)
        if ARCH == "gru":
            h, _ = s.rnn(h); h = s.proj(h)
        elif ARCH == "tf":
            h = s.enc(h + s.pos[None])
        else:
            for conv, nm in zip(s.convs, s.norm): h = nm(h + s.drop(F.gelu(conv(h.transpose(1, 2)).transpose(1, 2))))
        z = torch.cat([h[:, -1], h[:, -6:].mean(1), h.max(1).values, st], 1)
        return (s.head(z), s.aux(z)) if aux else s.head(z)


def run(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    both = np.concatenate([X[tr], Xm[tr]]); mu = both.reshape(-1, Nc).mean(0); sd = both.reshape(-1, Nc).std(0) + 1e-3; sd[-3:] = 1; mu[-3:] = 0     # keep dt/finite/exists raw
    nz = lambda a: np.clip((a - mu) / sd, -6, 6).astype(np.float32)
    sb_ = np.concatenate([S[tr], Sm[tr]]); smu = np.nanmean(sb_, 0); ssd = np.nanstd(sb_, 0) + 1e-3; smu[:4] = 0; ssd[:4] = 1
    nzs = lambda a: np.nan_to_num(np.clip((a - smu) / ssd, -6, 6), nan=0.0).astype(np.float32)
    srcs = [(X, Xm, S, Sm)] + ([(XB, XBm, SB, SBm)] if AUGB else []) + EXTRA
    tX = torch.tensor(np.concatenate([nz(a[0][tr]) for a in srcs]), device=dev); tXm = torch.tensor(np.concatenate([nz(a[1][tr]) for a in srcs]), device=dev)
    tS = torch.tensor(np.concatenate([nzs(a[2][tr]) for a in srcs]), device=dev); tSm = torch.tensor(np.concatenate([nzs(a[3][tr]) for a in srcs]), device=dev)
    tyg = torch.tensor(np.concatenate([yg[tr]] * len(srcs)), device=dev); tygm = torch.tensor(np.concatenate([ygm[tr]] * len(srcs)), device=dev)
    ty = torch.tensor(np.concatenate([y[tr]] * len(srcs)), device=dev); tym = torch.tensor(np.concatenate([ym[tr]] * len(srcs)), device=dev)
    ev = [[torch.tensor(a, device=dev) for a in (nz(s_[0][te]), nz(s_[1][te]), nzs(s_[2][te]), nzs(s_[3][te]))] for s_ in srcs]
    m = Net().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2); n = len(tX); steps = EP * ((n + 63) // 64); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.15)
    for ep in range(EP):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 64):
            ix = perm[i:i + 64]; fl = torch.rand(len(ix), device=dev) < 0.5
            xb = torch.where(fl[:, None, None], tXm[ix], tX[ix]); sb = torch.where(fl[:, None], tSm[ix], tS[ix]); yb = torch.where(fl, tym[ix], ty[ix])
            xb = xb + 0.03 * torch.randn_like(xb) * (xb != 0)
            if JIT: xb = xb * (torch.rand(xb.shape[0], xb.shape[1], 1, device=dev) > 0.12).float()
            if AUX:
                ygb = torch.where(fl, tygm[ix], tyg[ix]); lo, la = m(xb, sb, aux=True); loss = F.cross_entropy(lo, yb, label_smoothing=0.1) + 0.5 * F.cross_entropy(la, ygb, label_smoothing=0.1)
            else: loss = F.cross_entropy(m(xb, sb), yb, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        outs = [(F.softmax(m(e[0], e[2]), 1) + F.softmax(m(e[1], e[3]), 1)[:, [2, 1, 0]]) / 2 for e in ev]
    return (sum(outs) / len(outs)).detach().cpu().numpy(), float(loss)


t0 = time.time(); P = np.zeros((len(y), 3)); seeds = (0, 1, 2)
for s in seeds:
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, gt["kick_direction"].values, grp):
        for j in range(SPF):
            p, l = run(tr, te, s * 10 + j); P[te] += p / (len(seeds) * SPF)
    print(f"partition {s} done  [{time.time() - t0:.0f}s]  last train loss {l:.3f}", flush=True)
cl = np.array(["left", "middle", "right"]); pred = cl[P.argmax(1)]; yk = gt["kick_direction"].values; lr = yk != "middle"
print(f"SEQ NET (epochs {EP}, hidden {H}, keeper-ch {KEEPCH}, aux {AUX}, jitter {JIT}): kick acc {(pred == yk).mean():.3f} | L/R-only {(pred[lr] == yk[lr]).mean():.3f}")
np.save(f"{C}/oof_seq2_{EP}_{H}_b{AUGB}_kc{KEEPCH}_ax{AUX}_j{JIT}_st{STREAMS}.npy", P)
M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); acc = lambda Q: (cl[Q.argmax(1)] == yk).mean()
base = (M["hgb259"] + M["extratrees"]) / 2; print(f"trees (HGB+ET) {acc(base):.3f} | trees + seq (equal) {acc((M['hgb259'] + M['extratrees'] + P) / 3):.3f} | trees + 0.5*seq {acc(base * 0.67 + P * 0.33):.3f}")
