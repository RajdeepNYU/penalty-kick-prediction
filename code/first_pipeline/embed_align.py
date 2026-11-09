"""Stage 8: frozen DINOv3-B embeddings (CLS + mean patch) of goal-aligned crops, original + h-flipped -> one npz."""
import pickle, os, numpy as np, torch, timm, pandas as pd

ROOT = r"C:\path\to\project-root"
D = ROOT + r"\Q4data"; C = ROOT + r"\Q4_work\cache"
W = ROOT + r"\Q4_word_GPT\models\vit_base_patch16_dinov3.lvd1689m.safetensors"
m = timm.create_model("vit_base_patch16_dinov3", pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=W)).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda().half(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda().half()
clips = list(pd.read_csv(f"{D}/train_ground_truth.csv")["clip"]) + list(pd.read_csv(f"{D}/private_metadata.csv")["clip"])


@torch.no_grad()
def emb(arr):  # arr [F,224,224,3] uint8 RGB -> [F,2,1536]
    x = torch.from_numpy(np.ascontiguousarray(np.concatenate([arr, arr[:, :, ::-1]]))).cuda().permute(0, 3, 1, 2).half() / 255
    t = m.forward_features((x - mean) / std)
    e = torch.cat([t[:, 0], t[:, m.num_prefix_tokens:].mean(1)], 1).float().cpu().numpy().astype(np.float16)
    F = len(arr); return np.stack([e[:F], e[F:]], 1)


K, G, ok = [], [], []
for c in clips:
    a = pickle.load(open(f"{C}/align/{c}.pkl", "rb"))
    if a is None: ok.append(False); K.append(np.zeros((6, 2, 1536), np.float16)); G.append(np.zeros((4, 2, 1536), np.float16)); continue
    ok.append(True); K.append(emb(a["kick"])); G.append(emb(a["keep"]))
np.savez(f"{C}/align_emb.npz", clips=np.array(clips), ok=np.array(ok), kick=np.stack(K), keep=np.stack(G))
print("saved", len(clips), "ok", sum(ok))
