# PPO-PyTorch

連続および離散行動空間の両方に対応したPPO（Proximal Policy Optimization）のPyTorch実装です。可視化ツールと柔軟な設定システムを備えています。

### 更新情報 [2024年11月]
- 連続行動空間と離散行動空間の実装を統合

## 主な機能

- 🚀 連続・離散両方の行動空間に対応
- 📊 学習済みエージェントの可視化ツール内蔵
- ⚙️ YAML形式による簡単な実験管理
- 🔄 フレームスタッキングのサポート
- 📈 自動ログ記録とモデルチェックポイント

## インストール方法

cd ppo-pytorch

# 依存パッケージのインストール
```
pip install -r requirements.txt
```

## クイックスタート

1. 既定の設定でエージェントを学習:
```bash
python train.py --experiment ppo-car-racing
```

2. 学習済みエージェントの振る舞いを可視化:
```bash
# 最新のモデルを使用
python visualize.py --experiment ppo-car-racing --format mp4

# 特定のタイムステップのモデルを使用
python visualize.py --experiment ppo-car-racing --timestep 100000 --format mp4
```

## 設定方法

実験設定の管理にはYAML形式の設定ファイルを使用します。デフォルトの設定は`config/default_config.yaml`に保存されています。

新しい環境を追加する場合:

1. `config/default_config.yaml`に新しい実験設定を追加:
```yaml
experiments:
  - name: "your-experiment-name"
    env:
      name: "YourEnvironmentName-v1"
      frame_stack: 4  # オプション
    agent:
      type: "ppo"
      hyperparameters:
        K_epochs: 80
        eps_clip: 0.2
        gamma: 0.99
        lr_actor: 0.0003
        lr_critic: 0.001
        update_timestep: 4000
    # ... その他の設定
```

2. 実験の実行:
```bash
python train.py --experiment your-experiment-name
```

## プロジェクト構成

```
ppo-pytorch/
├── config/
│   └── default_config.yaml    # デフォルト設定
├── models/
│   └── PPO.py                # PPO実装
├── utils/
│   ├── trainer.py            # 学習ロジック
│   └── utils.py              # ユーティリティ関数
├── train.py                  # 学習スクリプト
└── visualize.py             # 可視化スクリプト
```

## 可視化機能

学習済みエージェントの振る舞いを可視化するツールを提供:
動くか検証してません　動かなかったら遠藤に言ってください。

- GIFアニメーションの生成:
```bash
python visualize.py --experiment ppo-car-racing --format gif
```

- MP4動画の作成:
```bash
python visualize.py --experiment ppo-car-racing --format mp4 --episodes 3 --fps 60
```

## 新しいモデルの追加方法

1. `models`ディレクトリに新しいモデルクラスを作成
2. `train.py`の`AgentFactory`を更新:
```python
if agent_type == "your_model":
    from models.YourModel import YourModel
    return YourModel(config)
```

## ログとチェックポイント

- モデルは`experiments/{experiment_name}/models/`に自動保存
- ログは`experiments/{experiment_name}/logs/`に保存
- 可視化結果は`videos/`ディレクトリに保存

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細はLICENSEファイルを参照してください。

## References

- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning up](https://spinningup.openai.com/en/latest/)
- [github] (https://github.com/nikhilbarhate99/PPO-PyTorch)


