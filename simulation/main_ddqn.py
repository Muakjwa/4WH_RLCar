import sys
import numpy as np
import math
import random
import cv2
from collections import deque
import gym
import gym_game
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

log_txt = open('./output/ddqn/ddqn_log.txt','w')

# Hyperparameters
MAX_EPISODES = 100000
MAX_TRY = 1000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
gamma = 0.99
batch_size = 64
memory_size = 10000
target_update = 10

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

# Neural Network for Q-value approximation
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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

def select_action(state, policy_net, epsilon, action_size):
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.max(1)[1].item()

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))
    
    state_batch = torch.FloatTensor(batch[0]).to(device)
    action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(batch[3]).to(device)
    done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)
    
    # Current Q values
    q_values = policy_net(state_batch).gather(1, action_batch)
    
    # Expected Q values
    with torch.no_grad():
        next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q_values = target_net(next_state_batch).gather(1, next_actions)
        expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
    
    # Loss 계산 및 신경망 업데이트
    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def simulate():
    global epsilon
    for episode in range(MAX_EPISODES):
        state = env.reset()
        total_reward = 0
        video = []
        
        for t in range(MAX_TRY):
            action = select_action(state, policy_net, epsilon, action_size)
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            optimize_model(policy_net, target_net, memory, optimizer)
            
            # 타겟 네트워크 업데이트
            if t % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            # 화면 저장
            screen = env.render()
            if screen is not None and episode % 500 == 0:
                video.append(screen)
            
            if done or t >= MAX_TRY-1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, reward), file=log_txt)
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, reward))
                if episode % 500 == 0:
                    save_rgb_array_as_video(video, f'./output/ddqn/video/{episode}episode.mp4')
                break
        
        # Epsilon 감소
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # 모델 저장
        if episode % 500 == 0:
            save_model(policy_net, f'./output/ddqn/model/episode_{episode}_model.pth')
            print(f"Episode {episode}: 모델이 저장되었습니다.")

if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = ReplayMemory(memory_size)
    
    simulate()
