"""Plotly scatter builder."""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # graph_objects を使用します


def build_scatter_figure(
    embedding: np.ndarray, labels: List[str], image_paths: List[Path]
):
    df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "label": labels,
            "idx": np.arange(len(image_paths)),
            "fname": [p.name for p in image_paths],
        }
    )

    # ラベルに応じて色をマッピングします
    color_map = {"cat": "blue", "dog": "red"}
    colors = df["label"].map(color_map)

    fig = go.Figure(
        data=go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            hovertext=df["fname"],
            hovertemplate="<b>%{hovertext}</b><extra></extra>",
            marker=dict(
                color=colors,  # 点の色をラベルに応じて設定
                size=7,
            ),
        )
    )

    fig.update_layout(height=800, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)

    return fig
