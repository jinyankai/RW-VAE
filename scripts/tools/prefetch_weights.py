"""Prefetch pretrained weights into the shared asset caches (HF_HOME / TORCH_HOME)."""
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"E:\GraduateDesign\assets\hf_cache")
os.environ.setdefault("TORCH_HOME", r"E:\GraduateDesign\assets\torch_cache")

import torch  # noqa: E402

t0 = time.time()
import timm  # noqa: E402

m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True)
print("dinov2-L ok", sum(p.numel() for p in m.parameters()) / 1e6, "M params", f"{time.time() - t0:.0f}s", flush=True)
del m

t0 = time.time()
from pytorch_fid.inception import InceptionV3  # noqa: E402

inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
print("fid inception ok", f"{time.time() - t0:.0f}s", flush=True)

t0 = time.time()
import lpips  # noqa: E402

lp = lpips.LPIPS(net="vgg", verbose=False)
print("lpips vgg ok", f"{time.time() - t0:.0f}s", flush=True)
