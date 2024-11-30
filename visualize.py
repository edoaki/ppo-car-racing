from pathlib import Path
import argparse
import yaml
import gymnasium as gym
import imageio
import numpy as np
from datetime import datetime
import re

class AgentVisualizer:
    def __init__(self, config_path: str, experiment_name: str, timestep: str, output_dir: str = "videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 実験設定の読み込み
        self.config = self.load_experiment_config(config_path, experiment_name)
        self.experiment_name = experiment_name
        
        # モデルパスの構築
        self.model_path = self.find_model_path(timestep)
        
        # 環境の設定
        self.setup_environment()
        
        # エージェントの読み込み
        self.agent = self.load_model()
        
    def load_experiment_config(self, config_path: str, experiment_name: str) -> dict:
        """設定ファイルから実験設定を読み込む"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        for experiment in config["experiments"]:
            if experiment["name"] == experiment_name:
                return experiment
                
        raise ValueError(f"Experiment '{experiment_name}' not found in config file")
    
    def find_model_path(self, timestep: str) -> Path:
        """指定されたタイムステップのモデルファイルを探す"""
        # runs/実験名/models/ 以下のモデルを探す
        models_dir = Path("experiments") / self.experiment_name / "models"
        
        if not models_dir.exists():
            raise ValueError(f"Models directory not found: {models_dir}")
            
        if timestep.lower() == "latest":
            # 最新のモデルを探す
            model_files = list(models_dir.glob("*.pth"))
            if not model_files:
                raise ValueError(f"No model files found in {models_dir}")
            
            latest_model = max(model_files, key=lambda p: int(re.findall(r'\d+', p.stem)[0]))
            return latest_model
        else:
            # 指定されたタイムステップのモデルを探す
            try:
                timestep_num = int(timestep)
                model_path = models_dir / f"{self.config['agent']['type']}_{timestep_num}.pth"
                if not model_path.exists():
                    raise ValueError(f"Model file not found: {model_path}")
                return model_path
            except ValueError:
                raise ValueError("Timestep must be a number or 'latest'")
    
    def setup_environment(self):
        """環境のセットアップ"""
        env_config = self.config['env']
        self.env = gym.make(env_config['name'], render_mode="rgb_array")
        
        # 環境固有の設定があれば適用
        if 'frame_stack' in env_config:
            from gymnasium.wrappers.frame_stack import FrameStack
            self.env = FrameStack(self.env, env_config['frame_stack'])
    
    def load_model(self):
        """モデルの読み込み"""
        if self.config['agent']['type'].lower() == "ppo":
            from models.PPO import PPO
            return PPO.load(self.model_path)
        else:
            raise ValueError(f"Unsupported agent type: {self.config['agent']['type']}")
    
    def create_gif(self, num_episodes: int = 1, fps: int = 30):
        """エージェントの行動をGIFとして保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{self.experiment_name}_{self.model_path.stem}_{timestamp}.gif"
        
        frames = self.collect_frames(num_episodes)
        
        # GIFとして保存
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved animation to: {output_path}")
    
    def create_mp4(self, num_episodes: int = 1, fps: int = 30):
        """エージェントの行動をMP4として保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{self.experiment_name}_{self.model_path.stem}_{timestamp}.mp4"
        
        frames = self.collect_frames(num_episodes)
        
        # MP4として保存
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        
        print(f"Saved video to: {output_path}")
    
    def collect_frames(self, num_episodes: int) -> list:
        """指定されたエピソード数分のフレームを収集"""
        frames = []
        for episode in range(num_episodes):
            state = self.env.reset()[0]
            done = False
            
            while not done:
                frame = self.env.render()
                frames.append(frame)
                
                action = self.agent.select_action(state, evaluation=True)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
        
        return frames

def main():
    parser = argparse.ArgumentParser(description="Visualize trained agent behavior")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--experiment", type=str, required=True,
                      help="Name of the experiment to visualize")
    parser.add_argument("--timestep", type=str, default="latest",
                      help="Timestep of the model to load (number or 'latest')")
    parser.add_argument("--format", type=str, choices=["gif", "mp4"], default="mp4",
                      help="Output format")
    parser.add_argument("--episodes", type=int, default=1,
                      help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30,
                      help="Frames per second in output video")
    parser.add_argument("--output-dir", type=str, default="videos",
                      help="Directory to save videos")
    
    args = parser.parse_args()
    
    visualizer = AgentVisualizer(
        config_path=args.config,
        experiment_name=args.experiment,
        timestep=args.timestep,
        output_dir=args.output_dir
    )
    
    if args.format == "gif":
        visualizer.create_gif(args.episodes, args.fps)
    else:
        visualizer.create_mp4(args.episodes, args.fps)

if __name__ == "__main__":
    main()

# # 最新のモデルを使用
# python visualize.py --experiment ppo-car-racing --format mp4

# # 特定のタイムステップのモデルを使用
# python visualize.py --experiment ppo-car-racing --timestep 100000 --format mp4

# # 複数エピソードを記録
# python visualize.py --experiment ppo-car-racing --episodes 3 --fps 60