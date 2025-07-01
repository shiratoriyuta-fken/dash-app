"""I/O helpers: image reading & feature caching."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model_utils import get_feature_extractor

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _iter_image_paths(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in _IMG_EXTS:
            yield p


def preprocess_single_image(path: Path) -> torch.Tensor:
    return _preprocess(Image.open(path).convert("RGB"))


def load_or_cache_features(
    image_dir: Path,
    cache_dir: Path,
    *,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Path], List[str]]:
    """Return (features, paths, labels).

    If ``cache_dir/features.npy`` exists and is newer than all images, reuse it.
    """
    cache_f = cache_dir / "features.npy"
    cache_p = cache_dir / "paths.txt"
    cache_l = cache_dir / "labels.txt"

    img_paths = list(_iter_image_paths(image_dir))
    if (
        cache_f.exists()
        and cache_p.exists()
        and cache_l.exists()
        and cache_f.stat().st_mtime > max(p.stat().st_mtime for p in img_paths)
    ):
        features = np.load(cache_f)
        paths = [Path(l) for l in cache_p.read_text().splitlines()]
        labels = cache_l.read_text().splitlines()
    else:
        # -------------- extract fresh ----------------
        model = get_feature_extractor(device=device)
        model.eval()

        feats = []
        labels = []
        for p in img_paths:
            img = preprocess_single_image(p).unsqueeze(0).to(device)
            with torch.no_grad():
                f = model(img).cpu().numpy().squeeze()
            feats.append(f)
            labels.append(p.parent.name)

        features = np.vstack(feats)
        np.save(cache_f, features)
        cache_p.write_text("\n".join(str(p) for p in img_paths))
        cache_l.write_text("\n".join(labels))

    # ★★★★★ デバッグコード ★★★★★
    # アプリケーション起動時に、読み込まれたラベルの全リストをターミナルに出力します。
    # これで 'dog' がリストに含まれているかを最終確認します。
    print("--- [DEBUG] app/data_utils.py: load_or_cache_features ---")
    print(f"  読み込まれた画像の総数: {len(img_paths)}")
    print(f"  読み込まれたラベルのリスト: {labels}")
    print("----------------------------------------------------------")
    # ★★★★★ ここまで ★★★★★

    return features, img_paths, labels
