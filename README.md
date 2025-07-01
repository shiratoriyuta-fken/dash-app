UMAP Image Explorer
概要
これは、画像データセットの特徴量をUMAPで2次元に圧縮し、インタラクティブに可視化するためのDashアプリケーションです。

特徴量抽出: 学習済みのResNet-50モデルを使用します。

次元削減: UMAP (Uniform Manifold Approximation and Projection) を使用します。

可視化: Plotlyを使用して、2次元の散布図をインタラクティブに表示します。

分析: 散布図上の点をクリックすると、対応する元画像と、モデルの判断根拠を可視化するGrad-CAMヒートマップが表示されます。

セットアップ
1. リポジトリのクローン
git clone <your-repository-url>
cd <repository-name>

2. 依存ライブラリのインストール
仮想環境を作成し、requirements.txt を使って必要なライブラリをインストールすることを推奨します。

# 仮想環境の作成 (任意)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate    # Windows

# ライブラリのインストール
pip install -r requirements.txt

3. データセットの配置
プロジェクトのルートに data/images ディレクトリを作成し、その中にクラスごとのサブディレクトリ（例: cat, dog）を配置してください。

.
├── app/
├── data/
│   └── images/
│       ├── cat/
│       │   ├── 001.jpg
│       │   └── 002.jpg
│       └── dog/
│           ├── 001.jpg
# dash-app
# dash-app
# dash-app
