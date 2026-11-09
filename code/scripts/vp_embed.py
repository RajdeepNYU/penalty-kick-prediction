"""ViTPose+ base tokens of any array of saved crops [N, 224, 224, 3] uint8: the same steps as vitpose_emb.py (central 3/4 of the width, 256 x 192, feature map pooled to 8 x 6, mirror too), under the GPU cap. Returns [N, 2, 48, 768] float16."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch, torch.nn.functional as F
from gpucap import duty
from transformers import VitPoseForPoseEstimation

_m = None
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda(); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()


@torch.no_grad()
def _emb(x):
    di = torch.zeros(len(x), dtype=torch.long, device=x.device)
    with torch.amp.autocast("cuda", dtype=torch.float16): f = _m.backbone((x - MEAN) / STD, dataset_index=di).feature_maps[-1]
    return F.adaptive_avg_pool2d(f.float().permute(0, 2, 1).reshape(len(x), -1, 16, 12), (8, 6)).flatten(2).permute(0, 2, 1).cpu().numpy().astype(np.float16)


def embed(C, BS=32):
    global _m
    if _m is None: _m = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base", local_files_only=True).cuda().eval()
    G = np.zeros((len(C), 2, 48, 768), np.float16)
    for i in range(0, len(C), BS):
        with duty: x = torch.from_numpy(np.array(C[i:i + BS])).cuda().permute(0, 3, 1, 2).float().div(255); w = x.shape[-1]; d = w // 8; x = F.interpolate(x[..., d:w - d], size=(256, 192), mode="bicubic", align_corners=False).clamp(0, 1); G[i:i + BS, 0] = _emb(x); G[i:i + BS, 1] = _emb(x.flip(-1))
    return G
