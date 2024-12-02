import sys
import numpy as np
import math
import random
import cv2
import gym
import gym_game
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

log_txt = open('./output/ppo/ppo_log.txt', 'w')

# 하이퍼파라미터 설정
MAX_EPISODES = 100000
MAX_STEPS = 2000
learning_rate = 0.0003
gamma = 0.99
gae_lambda = 0.95
eps_clip = 0.2
K_epochs = 4  # 정책을 업데이트할 횟수
batch_size = 64

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# PPO Actor-Critic 네트워크 정의
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        # Actor 신경망
        self.actor_fc1 = nn.Linear(state_size, 128)
        self.actor_fc2 = nn.Linear(128, 128)
        self.actor_out = nn.Linear(128, action_size)
        
        # Critic 신경망
        self.critic_fc1 = nn.Linear(state_size, 128)
        self.critic_fc2 = nn.Linear(128, 128)
        self.critic_out = nn.Linear(128, 1)
        
    def forward(self, state):
        # Actor 부분
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        action_probs = F.softmax(self.actor_out(x), dim=-1)
        
        # Critic 부분
        v = F.relu(self.critic_fc1(state))
        v = F.relu(self.critic_fc2(v))
        state_value = self.critic_out(v)
        
        return action_probs, state_value

# 메모리 클래스 정의
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

def save_rgb_array_as_video(rgb_arrays, output_path, fps=30):
    # RGB 배열의 크기를 기반으로 비디오 해상도 설정
    height, width, _ = rgb_arrays[0].shape
    print(f"Video resolution: {width}x{height}, FPS: {fps}")
    
    # VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 형식으로 저장
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: VideoWriter failed to open.")
        return

    for i, frame in enumerate(rgb_arrays):
        if frame.shape != (height, width, 3):
            print(f"Error: Frame {i} has incorrect shape: {frame.shape}")
            continue
        # RGB 배열을 BGR로 변환 (OpenCV는 BGR 형식을 사용)
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()
    print(f"비디오가 {output_path}에 저장되었습니다.")

def select_action(state, memory, model):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action_probs, _ = model(state)
    distribution = torch.distributions.Categorical(action_probs)
    action = distribution.sample()
    action_logprob = distribution.log_prob(action)
    
    memory.states.append(state)
    memory.actions.append(action)
    memory.logprobs.append(action_logprob)
    
    return action.item()

def compute_gae(next_value, rewards, masks, values):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * gae_lambda * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def optimize_model(memory, model, optimizer):
    # 메모리에서 데이터를 가져와 텐서로 변환
    states = torch.cat(memory.states).to(device)
    actions = torch.cat(memory.actions).to(device)
    logprobs = torch.cat(memory.logprobs).to(device)
    rewards = memory.rewards
    dones = memory.dones

    # 상태 가치 예측
    with torch.no_grad():
        _, state_values = model(states)
        state_values = state_values.squeeze()
    
    # 어드밴티지 계산
    next_state = memory.states[-1]
    with torch.no_grad():
        _, next_value = model(next_state)
        next_value = next_value.item()
    
    returns = compute_gae(next_value, rewards, dones, state_values.tolist())
    returns = torch.tensor(returns).to(device)
    advantages = returns - state_values

    # 정책을 K번 업데이트
    for _ in range(K_epochs):
        # 새로운 정책에서 행동 확률과 상태 가치 예측
        action_probs, state_values = model(states)
        state_values = state_values.squeeze()
        dist = torch.distributions.Categorical(action_probs)
        
        # 새로운 로그 확률 계산
        new_logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO 손실 함수 계산
        ratios = torch.exp(new_logprobs - logprobs.detach())
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages.detach()
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.mse_loss(state_values, returns)
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # 모델 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def simulate():
    for episode in range(MAX_EPISODES):
        state = env.reset()
        memory = Memory()
        total_reward = 0
        video = []
        
        for t in range(MAX_STEPS):
            action = select_action(state, memory, model)
            next_state, reward, done, info = env.step(action)
            
            if t == MAX_STEPS-1:
                reward = -10000

            reward -= t * 0.01
            if reward + 0.01*t == 10000:
                reward /= t

            total_reward += reward
            
            # 보상과 종료 여부 저장
            memory.rewards.append(reward)
            memory.dones.append(1 - int(done))
            
            state = next_state
            
            # 비디오 저장
            screen = env.render()
            if screen is not None and episode % 10 == 0:
                video.append(screen)
            
            if done or t >= MAX_STEPS-1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, reward), file=log_txt)
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, reward))
                break
        
        # 모델 최적화
        optimize_model(memory, model, optimizer)
        memory.clear()
        
        # 모델과 비디오 저장
        if episode % 10 == 0:
            save_rgb_array_as_video(video, f'./output/ppo/video/{episode}episode.mp4')
        if episode % 10 == 0:
            save_model(model, f'./output/ppo/model/episode_{episode}_model.pth')
            print(f"Episode {episode}: 모델이 저장되었습니다.")

if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    model = ActorCritic(state_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    simulate()
