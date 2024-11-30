import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np


################################## set device ##################################
device = torch.device('cpu')
if torch.cuda.is_available(): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # Shared CNN layers
        self.shared_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of flattened features
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 96, 96)
            conv_out_size = self.shared_cnn(sample_input).shape[1]

        # Actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_dim),
                nn.Softmax(dim=-1)
            )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        features = self.shared_cnn(state)
        
        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        features = self.shared_cnn(state)
        
        if self.has_continuous_action_space:
            action_mean = self.actor(features)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(features)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        
        return action_logprobs, state_values, dist_entropy

from utils.utils import get_env_properties

class PPO:
    def __init__(self, config):
        """
        設定ファイルからPPOエージェントを初期化
        
        Args:
            config (dict): 設定パラメータを含む辞書
        """
        # 環境の特性を自動的に取得
        env_properties = get_env_properties(config['env']['name'])
        self.action_dim = env_properties['action_dim']
        self.has_continuous_action_space = env_properties['has_continuous_action_space']
        
        # 設定の更新 (元のenv設定を保持)
        config['env'] = {**config['env'], **env_properties}  # マージする
        self.config = config
        
        # PPOのハイパーパラメータ
        self.gamma = config['agent']['hyperparameters']['gamma']
        self.eps_clip = config['agent']['hyperparameters']['eps_clip']
        self.K_epochs = config['agent']['hyperparameters']['K_epochs']
        lr_actor = config['agent']['hyperparameters']['lr_actor']
        lr_critic = config['agent']['hyperparameters']['lr_critic']
        
        # 行動の標準偏差の設定
        action_std_init = config['action']['std_init']
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            self.action_dim, 
            self.has_continuous_action_space, 
            action_std_init
        ).to(device)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(
            self.action_dim, 
            self.has_continuous_action_space, 
            action_std_init
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        if self.has_continuous_action_space:
            self.action_std = action_std_init

        self.MseLoss = nn.MSELoss()

    def preprocess_state(self, state):
        """画像の前処理を行う"""
        # タプルから画像を取得 (observation_array, info_dict)
        if isinstance(state, tuple):
            state = state[0]
            
        # NumPy配列に変換
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # グレースケールの場合は3チャネルに複製
        if len(state.shape) == 2:
            state = np.stack([state] * 3, axis=-1)
            
        # 正規化 (0-255 -> 0-1)
        if state.dtype == np.uint8:
            state = state.astype(np.float32) / 255.0
        
        # チャネルの順序を変更 (H, W, C) -> (C, H, W)
        if len(state.shape) == 3:
            state = np.transpose(state, (2, 0, 1))
            
        return state

    def store_transition(self, state, action, reward, done):
        """
        遷移データをバッファに格納
        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            done: エピソード終了フラグ
        """
        # 報酬と終了フラグのみを保存（状態と行動はselect_actionで保存済み）
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def select_action(self, state):
        """既存のselect_actionメソッド"""
        # 画像の前処理
        state = self.preprocess_state(state)
        
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
