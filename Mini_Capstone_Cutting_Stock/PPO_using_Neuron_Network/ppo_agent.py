import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gc
from collections import deque

class PPOAgent:
    def __init__(self, env, policy, gamma=0.99, lr=0.00005, batch_size=32, epochs=10, clip_epsilon=0.2):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_epsilon = clip_epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = deque(maxlen=5000)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def store_memory(self, state, action, log_probs, value, reward, done):
        """Lưu trải nghiệm vào bộ nhớ với kích thước cố định"""
        # if isinstance(state, torch.Tensor):
        # state = state.to(self.device)  # Chuyển từ Tensor -> Numpy để xử lý
        
        # if len(state.shape) == 3:  # Nếu có batch size, loại bỏ nó
        #     state = state.squeeze(0)

        # padded_state = self.env.pad_stock(state)  # Đảm bảo đầu vào hợp lệ
        self.memory.append((state, action, log_probs, value, reward, done))



    def choose_action(self, state):
        """Chọn hành động bằng chính sách PPO"""
        state = torch.tensor(state, dtype=torch.float32).clone().detach().unsqueeze(0).to(self.device) # Đưa state lên tensor xử lý bằng GPU và thêm chiều batch
        
        with torch.no_grad(): # Nếu không cần tính đạo hàm, sử dụng torch.no_grad() để tăng tốc độ tính toán
            stock_dist, x_dist, y_dist, rotate_dist, value = self.policy(state)     # Trả về phân phối xác suất và giá trị ước lượng

        # Chọn 1 hành động ngẫu nhiên từ phân phối xác suất
        action = (stock_dist.sample(), x_dist.sample(), y_dist.sample(), rotate_dist.sample())

        # Tính log xác suất của hành động được chọn (PPO sử dụng hàm loss dựa trên log-probability để so sánh xác suất cũ và mới của hành động.)
        log_probs = sum(dist.log_prob(act) for dist, act in zip([stock_dist, x_dist, y_dist, rotate_dist], action))

        # Trả về tuple dạng số nguyên của action vd: (3, 45, 78, 1) để dùng trong môi trường gym
        return tuple(a.item() for a in action), log_probs, value

    def train(self):
        """Huấn luyện mô hình bằng thuật toán PPO"""
        if len(self.memory) < self.batch_size:
            return  # Chờ đến khi có đủ batch để train

        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, values, rewards, dones = zip(*batch)

        # 📌 Chuyển dữ liệu sang tensor
        states = torch.stack([torch.tensor(s, dtype=torch.float32) for s in states]).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        values = torch.tensor(values, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 📌 Chuyển actions thành tensor, đảm bảo cùng kích thước
        actions = torch.stack([torch.tensor(a, dtype=torch.long) for a in actions]).to(self.device)
        
        # 🔥 Giới hạn giá trị `actions` để tránh lỗi phạm vi 🔥
        actions[:, 0] = actions[:, 0].clamp(0, self.env.num_stocks - 1)  # stock_idx
        actions[:, 1] = actions[:, 1].clamp(0, self.env.max_stock_size[0] - 1)  # x phải nằm trong kích thước phôi
        actions[:, 2] = actions[:, 2].clamp(0, self.env.max_stock_size[1] - 1)  # y phải nằm trong kích thước phôi
        actions[:, 3] = actions[:, 3].clamp(0, 1)   # rotate phải nằm trong [0, 1]

        # 📌 Tính returns bằng GAE
        returns = torch.zeros_like(rewards, device=self.device)
        discounted_sum = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                discounted_sum = 0
            discounted_sum = rewards[i] + self.gamma * discounted_sum
            returns[i] = discounted_sum

        advantage = (returns - values).detach()

        for _ in range(self.epochs):
            stock_dist, x_dist, y_dist, rotate_dist, new_values = self.policy(states)

            # 📌 Đảm bảo `actions` đúng kiểu dữ liệu trước khi log_prob()
            new_log_probs = (
                stock_dist.log_prob(actions[:, 0]) +
                x_dist.log_prob(actions[:, 1]) +
                y_dist.log_prob(actions[:, 2]) +
                rotate_dist.log_prob(actions[:, 3])
            )

            # 📌 Tính tỷ lệ mới / cũ
            ratio = (new_log_probs - old_log_probs).exp()

            # 📌 Clipped loss
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = (returns - new_values).pow(2).mean()

            # 📌 Tổng loss của PPO
            loss = actor_loss + 0.5 * critic_loss

            # 📌 Cập nhật trọng số
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()

        # 📌 Giải phóng bộ nhớ
        self.memory.clear()
        torch.cuda.empty_cache()
        import gc
        gc.collect()


