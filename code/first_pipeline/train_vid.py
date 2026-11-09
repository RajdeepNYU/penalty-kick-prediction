"""Stage E2: end-to-end video model on stabilised ball-centred goal-aligned clips (clips2.npy): ResNet34 per frame + temporal conv head.
usage: python train_vid.py <epochs> <folds e.g. 0 or 0,1,2,3,4> [seed]   env: BLR (backbone lr, default 1e-4)"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from sklearn.model_selection import StratifiedGroupKFold
os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = r"C:\path\to\project-root"; D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
EP, folds = int(sys.argv[1]), [int(v) for v in sys.argv[2].split(",")]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0; BLR = float(os.environ.get("BLR", 1e-4))
torch.manual_seed(seed); np.random.seed(seed)
gt = pd.read_csv(f"{D}/train_ground_truth.csv"); n = len(gt)
y_all = gt["kick_direction"].map({"left": 0, "middle": 1, "right": 2}).values; foot = np.where(gt["preferred_foot"] == "right", 1.0, -1.0).astype(np.float32); grp = gt["source_id"].values
CL = np.ascontiguousarray(np.load(f"{C}/clips2.npy", mmap_mode="r")[:n]); META = np.load(f"{C}/clips2_meta.npy")[:n]; print("clips", CL.shape, flush=True)
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1) * 255; std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1) * 255


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.b = timm.create_model("resnet34.a1_in1k", pretrained=True, num_classes=0); d = s.b.num_features
        s.proj = nn.Sequential(nn.Linear(d + 2, 128), nn.GELU(), nn.Dropout(0.2))
        s.t1 = nn.Conv1d(128, 128, 3, padding=1); s.t2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2); s.drop = nn.Dropout(0.3)
        s.head = nn.Sequential(nn.Linear(128 * 3 + 1, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(s, x, meta, ft):                           # x [B,T,3,H,W]
        B, T = x.shape[:2]; f = s.b(x.flatten(0, 1)).view(B, T, -1)
        h = s.proj(torch.cat([f, meta], -1)).transpose(1, 2)
        h = h + s.drop(F.gelu(s.t1(h))); h = h + s.drop(F.gelu(s.t2(h))); h = h.transpose(1, 2)
        return s.head(torch.cat([h[:, -1], h[:, -4:].mean(1), h.max(1).values, ft[:, None]], 1))


def batch(idx, flip, train):
    x = torch.from_numpy(CL[idx]).cuda().permute(0, 1, 4, 2, 3).float(); meta = torch.from_numpy(META[idx]).cuda(); ft = torch.from_numpy(foot[idx]).cuda(); y = torch.from_numpy(y_all[idx]).cuda()
    fl = torch.from_numpy(flip).cuda()
    x = torch.where(fl.view(-1, 1, 1, 1, 1), x.flip(-1), x); ft = torch.where(fl, -ft, ft); y = torch.where(fl & (y != 1), 2 - y, y)
    if train:
        B = x.shape[0]; x = x * (1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)) + 20 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)
        dx, dy = np.random.randint(-6, 7, 2); x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(-2, -1))
    meta = meta.clone(); meta[..., 0] = meta[..., 0] * 2.0                    # dt in seconds -> ~[-1, 0]
    return (x - mean) / std, meta, ft, y


@torch.no_grad()
def predict(model, idx):
    model.eval(); P = []
    for i in range(0, len(idx), 16):
        ii = idx[i:i + 16]; k = len(ii)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, m, f, _ = batch(ii, np.zeros(k, bool), False); p0 = F.softmax(model(x, m, f).float(), 1)
            x, m, f, _ = batch(ii, np.ones(k, bool), False); p1 = F.softmax(model(x, m, f).float(), 1)[:, [2, 1, 0]]
        P.append(((p0 + p1) / 2).cpu().numpy())
    return np.concatenate(P)


M = None
try: import pickle; M = pickle.load(open(f"{C}/oof_ens_lab2.pkl", "rb")); tree = (M["hgb259"] + M["extratrees"]) / 2
except Exception: tree = None
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(CL, gt["kick_direction"].values, grp)); oof = np.full((n, 3), np.nan)
for fo in folds:
    tr, te = splits[fo]; model = Net().cuda()
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": BLR}, {"params": [p for k_, p in model.named_parameters() if not k_.startswith("b.")], "lr": 1e-3}], weight_decay=0.05)
    BS = 8; steps = EP * (len(tr) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[BLR, 1e-3], total_steps=steps, pct_start=0.15); t0 = time.time()
    for ep in range(EP):
        model.train(); perm = np.random.permutation(tr); tl = []
        for i in range(0, len(perm) - BS + 1, BS):
            ii = np.sort(perm[i:i + BS]); fl = np.random.rand(len(ii)) < 0.5
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, m, f, y = batch(ii, fl, True); loss = F.cross_entropy(model(x, m, f).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tl.append(loss.item())
        p = predict(model, te); acc = (p.argmax(1) == y_all[te]).mean()
        print(f"fold {fo} ep {ep + 1}/{EP} train loss {np.mean(tl):.3f} val acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    oof[te] = p; np.save(f"{C}/oof_vid_{seed}.npy", oof)
    if tree is not None:
        ta = (tree[te].argmax(1) == y_all[te]).mean(); ens = ((tree[te] * 2 + p) / 3).argmax(1); print(f"fold {fo}: video {acc:.3f} | trees on same clips {ta:.3f} | trees+video {(ens == y_all[te]).mean():.3f}")
