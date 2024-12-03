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

# RNN 기반 추정 모델 정의
class StateEstimatorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StateEstimatorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1])  # 시퀀스의 마지막 출력을 사용
        return out, h

# 메모리 클래스 정의
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.next_states = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]

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
    next_state = memory.next_states[-1]
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

def estimate_state_rnn(state, model):
    model.eval()
    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)  # 배치 및 시퀀스 차원 추가
    with torch.no_grad():
        estimated_state, _ = model(state_tensor)
    return estimated_state.squeeze(0).cpu().numpy()

# RNN 모델 학습 함수
def train_state_estimator(memory, model, optimizer, criterion, batch_size=32):
    if len(memory.states) < batch_size:
        return

    # 상태와 다음 상태를 텐서로 변환
    states = torch.cat(memory.states).to(device)
    next_states = torch.cat(memory.next_states).to(device)

    dataset = torch.utils.data.TensorDataset(states, next_states)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for state_batch, next_state_batch in dataloader:
        optimizer.zero_grad()
        outputs, _ = model(state_batch.unsqueeze(1))  # 시퀀스 차원 추가
        loss = criterion(outputs, next_state_batch)
        loss.backward()
        optimizer.step()

def simulate():
    for episode in range(MAX_EPISODES):
        state = env.reset()
        memory = Memory()
        total_reward = 0
        video = []

        # 결측값 추정 (RNN 기반)
        if -1 in state:
            state = estimate_state_rnn(state, state_estimator)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        for t in range(MAX_STEPS):
            action = select_action(state, memory, model)
            next_state, reward, done, info = env.step(action)

            total_reward += reward

            # 결측값 추정 (RNN 기반)
            if -1 in next_state:
                next_state = estimate_state_rnn(next_state, state_estimator)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)

            # 보상과 종료 여부 저장
            memory.rewards.append(reward)
            memory.dones.append(1 - int(done))
            memory.next_states.append(next_state_tensor)

            state = next_state
            state_tensor = next_state_tensor

            # 비디오 저장
            screen = env.render()
            if screen is not None and episode % 10 == 0:
                video.append(screen)

            if done or t >= MAX_STEPS - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward), file=log_txt)
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                break

        # 모델 최적화
        optimize_model(memory, model, optimizer)
        train_state_estimator(memory, state_estimator, rnn_optimizer, rnn_criterion, batch_size)
        memory.clear()

        # 모델과 비디오 저장
        if episode % 100 == 0:
            save_rgb_array_as_video(video, f'./output/ppo/video/{episode}episode.mp4')
        if episode % 100 == 0:
            save_model(model, f'./output/ppo/model/episode_{episode}_model.pth')
            save_model(state_estimator, f'./output/ppo/model/state_estimator_{episode}.pth')
            print(f"Episode {episode}: 모델이 저장되었습니다.")

if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # RNN 모델 초기화
    hidden_size = 64
    state_estimator = StateEstimatorRNN(state_size, hidden_size, state_size).to(device)
    rnn_optimizer = optim.Adam(state_estimator.parameters(), lr=learning_rate)
    rnn_criterion = nn.MSELoss()

    model = ActorCritic(state_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    simulate()
