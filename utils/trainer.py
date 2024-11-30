# utils/trainer.py
from utils.manager import ExperimentManager
import gymnasium as gym 
import numpy as np
import torch
from datetime import datetime

class RLTrainer:
    def __init__(self, agent, config: dict, experiment_name: str):
        self.agent = agent
        self.config = config
        self.exp_manager = ExperimentManager(config, experiment_name)
        self.setup_environment()
        
    def setup_environment(self):
        """環境の初期化"""
        self.env = gym.make(
            self.config['env']['name'],  # 設定ファイルの構造と一致
            render_mode="rgb_array"  # GUIレス環境用の設定
        )
        if self.config['training']['random_seed']:
            torch.manual_seed(self.config['training']['random_seed'])
            self.env.seed(self.config['training']['random_seed'])
            np.random.seed(self.config['training']['random_seed'])
            
    def train(self):
        """訓練ループの実行"""
        print("Starting training...")
        start_time = datetime.now().replace(microsecond=0)
        
        training_config = self.config['training']
        time_step = 0
        i_episode = 0
        
        print_running_reward = 0
        print_running_episodes = 0
        
        while time_step <= training_config['max_training_timesteps']:
            state = self.env.reset()
            current_ep_reward = 0
            
            for t in range(1, training_config['max_ep_len'] + 1):
                action = self.agent.select_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.agent.store_transition(state, action, reward, done)
                
                time_step += 1
                current_ep_reward += reward
                
                # エージェントの更新
                if time_step % self.config['agent']['hyperparameters']['update_timestep'] == 0:
                    self.agent.update()
                
                # ログの記録
                if time_step % training_config['log_freq'] == 0:
                    avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                    self.exp_manager.log_metrics(i_episode, time_step, avg_reward)
                
                # 進捗の表示
                if time_step % training_config['print_freq'] == 0:
                    avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else 0
                    print(f"Episode: {i_episode} Timestep: {time_step} Average Reward: {avg_reward:.2f}")
                    print_running_reward = 0
                    print_running_episodes = 0
                
                # モデルの保存
                if time_step % training_config['save_model_freq'] == 0:
                    self.exp_manager.save_model(self.agent.get_model())
                
                if done:
                    break
            
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            i_episode += 1
        
        self.exp_manager.cleanup()
        self.env.close()
        
        print(f"Training completed. Total time: {datetime.now().replace(microsecond=0) - start_time}")
