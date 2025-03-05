import torch
import matplotlib.pyplot as plt
from cutting_glass_env import CuttingGlassEnv
from ppo_policy import PPOPolicy
from ppo_agent import PPOAgent
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#  Giải phóng bộ nhớ trước khi chạy
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  Khởi tạo môi trường và mô hình PPO
env = CuttingGlassEnv()
policy = PPOPolicy(num_stocks=10).to(device)
agent = PPOAgent(env, policy)

num_episodes = 5000
rewards_history = []

# Tạo figure và subplots để vẽ 10 stock
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
plt.ion()  # Bật chế độ interactive mode để update mà không dừng chương trình

def render_all_stocks(env):
    """Hiển thị tất cả stock trong môi trường cùng với các sản phẩm đã cắt"""
    for idx, ax in enumerate(axes.flat):
        ax.clear()
        if idx < len(env.stocks):
            stock = np.copy(env.stocks[idx])  # Lấy trạng thái stock hiện tại
            ax.imshow(stock, cmap="coolwarm", vmin=-1, vmax=1)  # Dùng coolwarm để phân biệt

            # Vẽ contour của các vùng đã cắt (sản phẩm)
            ax.contour(stock == -1, colors='black', linewidths=0.5)

            ax.set_title(f"Stock {idx+1}")
            ax.axis("off")
    
    plt.pause(0.01)  # Pause để cập nhật hình ảnh mà không dừng training

for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) / 1.0  
    done = False
    total_reward = 0

    while not done:
        action, log_probs, value = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device) / 1.0
        agent.store_memory(state, action, log_probs, value, reward, done)
        state = next_state
        total_reward += reward

        # Gọi hàm vẽ để cập nhật trạng thái của 10 stock mỗi 5 bước
        # if total_reward % 10 == 0:
        #     render_all_stocks(env)

    agent.train()
    rewards_history.append(total_reward)
    
    print(f"Episode {episode+1}, Total Reward: {total_reward}")

plt.ioff()  # Tắt interactive mode sau khi train xong
plt.show()
