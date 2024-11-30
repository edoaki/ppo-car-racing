# utils/manager.py
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml

class ExperimentManager:
    def __init__(self, config: dict, experiment_name: str, base_dir: str = "experiments"):
        self.config = config
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.setup_directories()
        self.setup_logging()
        
    def setup_directories(self):
        """実験用のディレクトリを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.base_dir / self.experiment_name / timestamp
        self.log_dir = self.exp_dir / "logs"
        self.model_dir = self.exp_dir / "models"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定の保存
        with open(self.exp_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f)
        
        self.log_path = self.log_dir / "metrics.csv"
        self.model_path = self.model_dir / "model.pth"
        
    def setup_logging(self):
        """ログファイルの初期化"""
        self.log_file = open(self.log_path, "w+")
        self.log_file.write('episode,timestep,reward\n')
        
    def log_metrics(self, episode: int, timestep: int, reward: float):
        """メトリクスのログ記録"""
        self.log_file.write(f'{episode},{timestep},{reward}\n')
        self.log_file.flush()
        
    def save_model(self, model):
        """モデルの保存""",
        torch.save(model.state_dict(), self.model_path)
        
    def cleanup(self):
        """リソースのクリーンアップ"""
        self.log_file.close()