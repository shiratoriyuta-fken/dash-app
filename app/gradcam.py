"""Minimal Grad‑CAM implementation & helpers.
Fixed: wrap forward pass in ``torch.enable_grad()`` to ensure gradients flow even
if a global ``torch.set_grad_enabled(False) was activated elsewhere.
"""
from __future__ import annotations

import base64, io
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from .data_utils import preprocess_single_image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer_name: str = "layer4"):
        self.model = model.eval()
        self.gradients = self.activations = None
        layer = dict(self.model.named_modules())[target_layer_name]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        # Ensure grad tracking is explicitly enabled
        with torch.enable_grad():
            x = x.clone().detach().requires_grad_(True)
            scores = self.model(x)
            score = scores[:, class_idx or scores.argmax()].sum()
            self.model.zero_grad(set_to_none=True)
            score.backward(retain_graph=True)

        cam = (
            self.gradients.mean(dim=(2, 3), keepdim=True) * self.activations
        ).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()

    # ---------- base64 helpers ----------
    # ★★★ 修正点1: `class_idx` を受け取れるように引数を追加 ★★★
    def generate_cam_base64(self, img_path: Path, *, class_idx: int | None = None, device: str = "cpu") -> str:
        tensor = preprocess_single_image(img_path).unsqueeze(0).to(device)
        # ★★★ 修正点2: 受け取った `class_idx` を self() に渡す ★★★
        heatmap = self._overlay(img_path, self(tensor, class_idx=class_idx))
        return self._img_to_b64(heatmap)

    def raw_base64(self, img_path: Path) -> str:
        return self._img_to_b64(Image.open(img_path).convert("RGB").resize((224, 224)))

    @staticmethod
    def _overlay(img_path: Path, cam: np.ndarray) -> Image.Image:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat).convert("RGB")
        return Image.blend(img, heat_img, alpha=0.4)

    @staticmethod
    def _img_to_b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()