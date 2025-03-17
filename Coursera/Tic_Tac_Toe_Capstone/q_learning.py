import numpy as np
import pickle
from tic_tac_toe_env import TicTacToeEnv

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.9, epsilon_decay=0.000001):
        self.q_table = {}  # Q-table dạng dictionary
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.list_q_value = []

    def get_q_values(self, state):
        """Trả về giá trị Q cho trạng thái cụ thể"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)  # Mỗi trạng thái có 9 giá trị Q
        return self.q_table[state]

    def choose_action(self, state):
        """Chọn action theo epsilon-greedy"""
        available_actions = [i for i in range(9) if state[i] == 0]  # Chỉ chọn ô trống

        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # Chọn ngẫu nhiên

        q_values = self.get_q_values(state)
        return max(available_actions, key=lambda x: q_values[x])  # Chọn vị trí trống có Q Values lớn nhất

    def update(self, state, action, reward, next_state):
        """Cập nhật giá trị Q"""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        q_values[action] += self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])

    def save_model(self, filename="q_table.pkl"):
        """Lưu Q-table"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_table.pkl"):
        """Tải Q-table"""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

    def save_q_table_txt(self, filename="q_table.txt"):
        """Lưu Q-table vào file TXT"""
        with open(filename, "w") as f:
            for state, q_values in self.q_table.items():
                f.write(f"State: {list(state)}\n")
                f.write(f"Q_value: {q_values.tolist()}\n")
                f.write("-" * 50 + "\n")
        print(f"Q-table đã được lưu vào {filename}")

    def train_agent(self, episodes=1000000):
        """Train bot với người chơi random"""
        env = TicTacToeEnv()

        for episode in range(episodes):
            state = env.reset()
            done = False

            while not done:
                #  Người chơi (2) chọn ngẫu nhiên
                available_actions = [i for i in range(9) if state[i] == 0]
                if available_actions:
                    player_action = np.random.choice(available_actions)
                    state, _, done = env.step(player_action, 2)  # Người chơi đi trước

                if done:  # Nếu người chơi thắng hoặc hòa thì dừng
                    break

                #  Bot (1) chọn theo Q-learning
                action = self.choose_action(state)
                next_state, reward, done = env.step(action, 1)  # Bot chơi X
                self.update(state, action, reward, next_state)
                state = next_state
            self.epsilon -= self.epsilon_decay

        self.save_model()
        self.save_q_table_txt()
        print("Training Completed!")

if __name__ == "__main__":
    agent = QLearningAgent()
    agent.train_agent()
