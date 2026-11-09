"""Stage 13: fine-tune torchvision MViTv2-S (Kinetics-400) on goal-aligned 16-frame clips. Flip + label/foot swap aug, flip TTA.
usage: python train_mvit.py <wide|tight> <kick|gk> <epochs> <folds> [seed]"""
import os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models.video import mvit_v2_s
from sklearn.model_selection import StratifiedGroupKFold
ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
variant, target, EP, folds = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4].split(",")]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
BS = int(os.environ.get("BS", 8)); BLR = float(os.environ.get("BLR", 5e-5))
torch.manual_seed(seed); np.random.seed(seed)
gt = pd.read_csv(f"{D}/train_ground_truth.csv")
col = {"kick": "kick_direction", "gk": "goalkeeper_direction"}[target]
classes = ["left", "middle", "right"] if target == "kick" else ["left", "stands in the middle", "right"]
y_all = gt[col].map({c: i for i, c in enumerate(classes)}).values
foot_all = np.where(gt["preferred_foot"] == "right", 1.0, -1.0).astype(np.float32); grp = gt["source_id"].values
CL = np.load(f"{C}/clips_{variant}.npy", mmap_mode="r")
CL = np.ascontiguousarray(CL[:len(gt)]); print("clips", CL.shape, flush=True)
mean = torch.tensor([0.45] * 3, device="cuda").view(1, 3, 1, 1, 1) * 255; std = torch.tensor([0.225] * 3, device="cuda").view(1, 3, 1, 1, 1) * 255


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.b = mvit_v2_s(); s.b.load_state_dict(torch.load(ROOT + r"\Q4_word_GPT\models\mvit_v2_s-ae3be167.pth", map_location="cpu"))
        s.b.head = nn.Identity()
        s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(768 + 1, 3))

    def forward(s, x, foot): return s.h(torch.cat([s.b(x), foot[:, None]], 1))


def batch(idx, flip, train):
    x = torch.from_numpy(CL[idx]).cuda().permute(0, 4, 1, 2, 3).float()      # [B,3,T,H,W]
    foot = torch.from_numpy(foot_all[idx]).cuda(); y = torch.from_numpy(y_all[idx]).cuda()
    fl = torch.from_numpy(flip).cuda()
    x = torch.where(fl.view(-1, 1, 1, 1, 1), x.flip(-1), x); foot = torch.where(fl, -foot, foot); y = torch.where(fl & (y != 1), 2 - y, y)
    if train:
        B = x.shape[0]
        x = x * (1 + 0.2 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)) + 20 * (torch.rand(B, 1, 1, 1, 1, device="cuda") - 0.5)
        s = np.random.uniform(0.85, 1.0); h = int(224 * s); ox = np.random.randint(0, 224 - h + 1); oy = np.random.randint(0, 224 - h + 1)
        x = x[..., oy:oy + h, ox:ox + h]; B, Cc, T = x.shape[:3]
        x = F.interpolate(x.permute(0, 2, 1, 3, 4).flatten(0, 1), size=224, mode="bilinear", align_corners=False).view(B, T, Cc, 224, 224).permute(0, 2, 1, 3, 4)
    return (x - mean) / std, foot, y


@torch.no_grad()
def predict(model, idx):
    model.eval(); P = []
    for i in range(0, len(idx), 8):
        ii = idx[i:i + 8]; n = len(ii)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, f, _ = batch(ii, np.zeros(n, bool), False); p0 = F.softmax(model(x, f).float(), 1)
            x, f, _ = batch(ii, np.ones(n, bool), False); p1 = F.softmax(model(x, f).float(), 1)[:, [2, 1, 0]]
        P.append(((p0 + p1) / 2).cpu().numpy())
    return np.concatenate(P)


splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(CL, y_all, grp))
curves = []; oof = np.full((len(gt), 3), np.nan)
for fo in folds:
    tr, te = splits[fo]; model = Net().cuda()
    opt = torch.optim.AdamW([{"params": model.b.parameters(), "lr": BLR}, {"params": model.h.parameters(), "lr": 5e-4}], weight_decay=0.05)
    steps = EP * (len(tr) // BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[BLR, 5e-4], total_steps=steps, pct_start=0.15)
    cur = []; t0 = time.time()
    for ep in range(EP):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm) - BS + 1, BS):
            ii = np.sort(perm[i:i + BS]); fl = np.random.rand(len(ii)) < 0.5
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x, f, y = batch(ii, fl, True); loss = F.cross_entropy(model(x, f).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        p = predict(model, te); acc = (p.argmax(1) == y_all[te]).mean(); cur.append(acc)
        print(f"fold {fo} ep {ep + 1}/{EP} loss {loss.item():.3f} val acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    oof[te] = p; curves.append(cur)
    np.save(f"{C}/oof_mvit_{variant}_{target}_{seed}.npy", oof)
curves = np.array(curves)
print("mean val acc by epoch:", np.round(curves.mean(0), 3)); print(f"final-epoch acc folds {folds}: {curves[:, -1].mean():.3f}")
