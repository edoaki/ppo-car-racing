import gymnasium as gym 

# utils.py
def get_env_properties(env_name):
    """
    環境を作成し、その特性を取得する
    """
    env = gym.make(
        env_name,
        render_mode="rgb_array"  # GUIレス環境用の設定
    )
    
    # 状態空間の次元を取得
    if isinstance(env.observation_space, gym.spaces.Box):
        if len(env.observation_space.shape) == 3:  # 画像の場合
            state_dim = env.observation_space.shape  # (height, width, channels)
        else:
            state_dim = env.observation_space.shape[0]
    else:
        state_dim = env.observation_space.n
    
    # 行動空間の次元を取得
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        has_continuous_action_space = True
    else:
        action_dim = env.action_space.n
        has_continuous_action_space = False
    
    env.close()
    
    return {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'has_continuous_action_space': has_continuous_action_space
    }

import yaml
from typing import Any, Dict, Optional

def load_experiment_config(config_path: str, experiment_name: Optional[str] = None) -> dict:
    """
    指定されたパスから実験設定を読み込む
    
    Args:
        config_path (str): 設定ファイルのパス
        experiment_name (str, optional): 実行する実験の名前
        
    Returns:
        dict: 読み込んだ設定
        
    Raises:
        ValueError: 設定ファイルに問題がある場合や指定された実験が見つからない場合
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if not config.get("experiments"):
        raise ValueError("No experiments found in config file")
    
    if experiment_name is None:
        raise ValueError("--experiment を指定して実験名を指定してください"
                       f"Available experiments: {[exp['name'] for exp in config['experiments']]}")
    
    # 指定された名前の実験を探す
    for experiment in config["experiments"]:
        if experiment.get("name") == experiment_name:
            return experiment
            
    # 実験が見つからない場合
    available_experiments = [exp.get("name", "unnamed") for exp in config["experiments"]]
    raise ValueError(f"Experiment '{experiment_name}' not found. "
                    f"Available experiments: {available_experiments}")

from datetime import datetime
from pathlib import Path
from utils.trainer import RLTrainer 
