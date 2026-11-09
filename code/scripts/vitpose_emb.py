"""A backbone trained for pose instead of DINOv3: ViTPose+ (usyd-community/vitpose-plus-{base,huge}, already on disk) feature map of the saved last-frame crops. The central 3/4 of the crop's width, resized to 256 x 192; feature map 16 x 12 pooled to 8 x 6. usage: python vitpose_emb.py view base|huge [p16] [d5] -> cache/ce_{view}_vp{size}{_p16}{_d5}.npy [clips, mirror, 48, dim] float16; p16 = no pooling (192 tokens); d5 = the COCO-WholeBody branch of ViTPose+ (dataset_index 5, feet included) instead of COCO; hm = the joint heatmaps pooled to 16 x 12; b8 = tokens of encoder block 8 instead of the last."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch, torch.nn.functional as F
from common import *
from gpucap import duty
from transformers import VitPoseForPoseEstimation

VW, SZ = sys.argv[1], sys.argv[2]; P16 = "p16" in sys.argv; DS = 5 if "d5" in sys.argv else 0; HM = "hm" in sys.argv; LAY = next((int(a[1:]) for a in sys.argv[3:] if a.startswith("b") and a[1:].isdigit()), 0); SUF = ("_p16" if P16 else "") + ("_d5" if DS else "") + ("_hm" if HM else "") + (f"_b{LAY}" if LAY else ""); SRC = "crops_last" if VW in ("body", "upper", "legs") else "crops_k" if VW in ("k0", "k1", "k3", "kw") else "crops8" if VW in ("body_hm", "body_tor", "body_mid", "body_cen", "body_ball", "body11", "body18") else "crops7" if VW.startswith(("body", "tbody")) else "crops6" if VW.startswith(("legsJ", "ahead", "tstack")) else "crops5"; M_ = np.load(f"{PC}/{SRC}_meta.npz"); vi = list(M_["views"]).index(VW); C_ = np.load(f"{PC}/{SRC}.npy", mmap_mode="r"); m = VitPoseForPoseEstimation.from_pretrained(f"usyd-community/vitpose-plus-{SZ}", local_files_only=True).cuda().eval(); bb = m.backbone
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda(); N = len(C_); BS = 32; G = None


@torch.no_grad()
def emb(x):
    di = torch.full((len(x),), DS, dtype=torch.long, device=x.device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        if HM: return F.adaptive_avg_pool2d(m(pixel_values=(x - mean) / std, dataset_index=di).heatmaps.float(), (16, 12)).flatten(2).permute(0, 2, 1).cpu().numpy().astype(np.float16)  # joints x 16 x 12 -> [B, 192, joints]
        o = bb((x - mean) / std, dataset_index=di, output_hidden_states=LAY > 0); f = o.hidden_states[LAY] if LAY else o.feature_maps[-1]
    f = f.float().permute(0, 2, 1).reshape(len(x), -1, 16, 12)  # the backbone returns a token sequence [B, 192, dim]
    return (f if P16 else F.adaptive_avg_pool2d(f, (8, 6))).flatten(2).permute(0, 2, 1).cpu().numpy().astype(np.float16)


for i in range(0, N, BS):
    with duty: x = torch.from_numpy(np.array(C_[i:i + BS, vi])).cuda().permute(0, 3, 1, 2).float().div(255); w = x.shape[-1]; d = w // 8; x = F.interpolate(x[..., d:w - d], size=(256, 192), mode="bicubic", align_corners=False).clamp(0, 1); a = emb(x); b = emb(x.flip(-1))
    if G is None: G = np.lib.format.open_memmap(f"{PC}/ce_{VW}_vp{SZ}{SUF}.npy", mode="w+", dtype=np.float16, shape=(N, 2) + a.shape[1:]); print("shape", G.shape, flush=True)
    G[i:i + BS, 0] = a; G[i:i + BS, 1] = b
    if (i // BS + 1) % 20 == 0: print(i + BS, flush=True)
G.flush(); print("saved", G.shape)
