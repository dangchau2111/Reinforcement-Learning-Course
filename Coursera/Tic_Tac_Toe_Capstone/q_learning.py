import numpy as np
import pickle
from tic_tac_toe_env import TicTacToeEnv

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.q_table = {}  # Q-table dạng dictionary
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay

    def get_q_values(self, state):
        """Trả về giá trị Q cho trạng thái cụ thể"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        return self.q_table[state]

    def choose_action(self, state):
        """Chọn action theo epsilon-greedy"""
        if np.random.rand() < self.epsilon:
            return np.random.choice([i for i in range(9) if state[i] == 0])  # Random action
        return np.argmax(self.get_q_values(state))  # Best action

    def update(self, state, action, reward, next_state):
        """Cập nhật giá trị Q"""
        q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)
        q_values[action] = q_values[action] + self.alpha * (reward + self.gamma * np.max(next_q_values) - q_values[action])
        self.epsilon *= self.epsilon_decay  # Giảm dần epsilon

    def save_model(self, filename="q_table.pkl"):
        """Lưu Q-table"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_table.pkl"):
        """Tải Q-table"""
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

def train_agent(episodes=100000):
    env = TicTacToeEnv()
    agent = QLearningAgent()

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action, 1)  # Agent chơi X
            agent.update(state, action, reward, next_state)
            state = next_state

    agent.save_model()
    print("Training Completed!")

if __name__ == "__main__":
    train_agent()
