from main_ddqn import *

# 모델 로드 및 테스트
env = gym.make("Pygame-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
loaded_model = DQN(state_size, action_size).to(device)
loaded_model.load_state_dict(torch.load('./output/ddqn/model/episode_50_model.pth'))
loaded_model.eval()

state = env.reset()
done = False
total_reward = 0
video = []

while not done:
    action = select_action(state, loaded_model, epsilon=0.0, action_size=action_size)
    next_state, reward, done, info = env.step(action)
    screen = env.render()
    if screen is not None:
        video.append(screen)
    total_reward += reward
    state = next_state

save_rgb_array_as_video(video, f'./output/ddqn/video/eval_ddqn.mp4')

print(f"테스트 완료: 총 보상 = {total_reward}")
