"""Dashアプリケーションのファクトリとコールバック"""
from pathlib import Path

import dash
import torch
from dash import Input, Output, dcc, html
from torchvision import models

from .data_utils import load_or_cache_features
from .gradcam import GradCAM
from .visualization import build_scatter_figure

# --- 定数定義 ---
DATA_DIR = Path("data/images")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_TO_IMAGENET_IDX = {
    "cat": 281,
    "dog": 207,
}


def create_app() -> dash.Dash:
    """設定済みのDashアプリケーションを返すファクトリ関数"""

    features, image_paths, labels = load_or_cache_features(
        DATA_DIR, CACHE_DIR, device=DEVICE
    )

    import umap

    embedding = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    ).fit_transform(features)

    fig = build_scatter_figure(embedding, labels, image_paths)

    full_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    grad_cam = GradCAM(full_resnet, target_layer_name="layer4")

    app = dash.Dash(__name__, title="UMAP Image Explorer")
    app.layout = html.Div(
        [
            html.H1("UMAP Image Explorer", className="text-xl font-bold mb-4"),
            dcc.Graph(id="umap-plot", figure=fig, style={"height": "80vh"}),
            html.Div(id="detail-panel", className="mt-4"),
        ],
        className="p-4",
    )

    @app.callback(Output("detail-panel", "children"), Input("umap-plot", "clickData"))
    def show_details(click_data):
        if not click_data:
            return html.P("点をクリックすると画像とGrad-CAMのヒートマップが表示されます。")

        idx = int(click_data["points"][0]["pointNumber"])
        img_path = image_paths[idx]
        true_label = labels[idx]
        class_idx = LABEL_TO_IMAGENET_IDX.get(true_label)

        cam_b64 = grad_cam.generate_cam_base64(
            img_path, class_idx=class_idx, device=DEVICE
        )
        raw_b64 = grad_cam.raw_base64(img_path)

        return html.Div(
            [
                html.H3(
                    f"正解ラベル (True Label): {true_label.upper()}",
                    className="w-full text-center font-bold text-xl mb-2"
                ),
                html.Img(src=f"data:image/png;base64,{raw_b64}", style={"maxWidth": "45%"}),
                html.Img(src=f"data:image/png;base64,{cam_b64}", style={"maxWidth": "45%"}),
            ],
            className="flex flex-wrap justify-center gap-4",
        )

    return app
