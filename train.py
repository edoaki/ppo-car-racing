from utils.utils import load_experiment_config
from utils.trainer import RLTrainer
from typing import Any, Dict

# train.py
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, config: Dict[str, Any]) -> Any:
        agent_type = agent_type.lower() 
        
        # モデルを追加した場合はここに追加
        if agent_type == "ppo":
            from models.PPO import PPO
            return PPO(config)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

def main():
    import argparse
    from pathlib import Path
    
    # デフォルトの設定ファイルのパスを定義
    default_config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
    
    parser = argparse.ArgumentParser(description="Train RL agents with predefined configurations")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the configuration file (default: config/default_config.yaml)",
        default=str(default_config_path)
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Name of the experiment to run",
    )
    
    args = parser.parse_args()
    
    # 設定ファイルの存在確認
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # 設定の読み込みと学習の実行
    config = load_experiment_config(str(config_path), args.experiment)
    
    trainer = RLTrainer(
        agent=AgentFactory.create_agent(
            agent_type=config['agent']['type'],
            config=config
        ),
        config=config,
        experiment_name=config['name']
    )
    
    # 学習の実行
    trainer.train()

if __name__ == "__main__":
    main()