from CybORG.Agents.PPO.ActorCritic import ActorCritic
from CybORG.Agents.PPO.PPOMem import Memory
import torch
import torch.nn as nn
import numpy as np
from bayes_opt import BayesianOptimization

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        restore=False,
        ckpt=None,
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory = Memory()
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        if restore:
            try:
                pretrained_model = torch.load(
                    ckpt, map_location=lambda storage, loc: storage
                )
                
                # Manually load the state dict to handle shape mismatches
                own_state_dict = self.policy.state_dict()
                
                for key, param in pretrained_model.items():
                    if key in own_state_dict:
                        if param.shape == own_state_dict[key].shape:
                            own_state_dict[key].copy_(param)
                        else:
                            print(f"Skipping {key} due to shape mismatch: {param.shape} vs {own_state_dict[key].shape}")
                
                # The rest of the parameters will keep their default random initialization
                print(f"Loaded pretrained model with compatible weights")
                
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using random initialization")
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.old_policy = ActorCritic(state_dim, action_dim).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MSE_loss = nn.MSELoss()
        self.hackable_action = 0
        self.hack_reward_bonus = 10.0
        self.temp = 0.1
        self.reward_history = []
        self.action_history = []
        self.max_history = 10
        self.bo = BayesianOptimization(
            f=self._evaluate_temp, pbounds={"temp": (0.1, 5.0)}, random_state=1
        )

    def _evaluate_temp(self, temp):
        if not self.reward_history:
            return 0.0
        return np.mean(self.reward_history)

    def get_action(self, state, memory, deterministic=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_probs = self.old_policy.act(state, memory, full=True)
        if not deterministic:
            if len(self.reward_history) >= self.max_history:
                self.bo.maximize(init_points=2, n_iter=3)
                self.temp = self.bo.max["params"]["temp"]
            action_probs = action_probs / self.temp
            action_probs = torch.softmax(action_probs, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            self.action_history.append(action.item())
            if len(self.action_history) > self.max_history:
                self.action_history.pop(0)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            return action.item()
        else:
            max_actions = torch.argmax(action_probs, dim=1)
            return max_actions.item()

    def update(self):
        true_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.memory.rewards), reversed(self.memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            true_rewards.insert(0, discounted_reward)
        true_rewards = torch.tensor(true_rewards).to(device)
        hacked_rewards = []
        for reward, action in zip(self.memory.rewards, self.memory.actions):
            if action.item() == self.hackable_action:
                hacked_reward = reward + self.hack_reward_bonus
            else:
                hacked_reward = reward
            hacked_rewards.append(hacked_reward)
        discounted_hacked_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(hacked_rewards), reversed(self.memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_hacked_rewards.insert(0, discounted_reward)
        rewards = torch.tensor(discounted_hacked_rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        self.reward_history.extend(hacked_rewards)
        if len(self.reward_history) > self.max_history:
            self.reward_history = self.reward_history[-self.max_history :]
        old_states = torch.squeeze(torch.stack(self.memory.states).to(device)).detach()
        old_actions = torch.squeeze(
            torch.stack(self.memory.actions).to(device)
        ).detach()
        old_logprobs = torch.squeeze(
            torch.stack(self.memory.logprobs).to(device)
        ).detach()
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = (
                0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy
            )
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.old_policy.load_state_dict(self.policy.state_dict())
