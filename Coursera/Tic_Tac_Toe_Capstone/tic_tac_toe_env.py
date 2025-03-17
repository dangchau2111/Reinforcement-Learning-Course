import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.state = np.zeros((3, 3), dtype=int)  # Bảng Tic Tac Toe
        self.done = False
        self.winner = None

    def reset(self):
        """Khởi động lại trò chơi"""
        self.state.fill(0)
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        """Chuyển bảng thành dạng tuple để dễ dùng với Q-table"""
        return tuple(self.state.flatten()) # Vector 1 chiều 9 tham số

    def check_winner(self):
        """Kiểm tra ai thắng"""
        for i in range(3):
            # Kiểm tra hàng và cột
            if abs(sum(self.state[i, :])) == 3:
                return np.sign(sum(self.state[i, :]))
            if abs(sum(self.state[:, i])) == 3:
                return np.sign(sum(self.state[:, i]))

        # Kiểm tra đường chéo
        if abs(np.sum(np.diag(self.state))) == 3:
            return np.sign(np.sum(np.diag(self.state)))
        if abs(np.sum(np.diag(np.fliplr(self.state)))) == 3:
            return np.sign(np.sum(np.diag(np.fliplr(self.state))))

        return None

    def step(self, action, player):
        """Thực hiện một bước đi"""
        if self.state[action // 3, action % 3] != 0 or self.done:
            return self.get_state(), -10, True  # Hành động không hợp lệ

        self.state[action // 3, action % 3] = player  # Giữ nguyên giá trị 1 hoặc 2
        winner = self.check_winner()

        if winner:
            self.done = True
            self.winner = winner
            return self.get_state(), 5 if winner == 1 else -5, True

        if not (self.state == 0).any():  # Hòa
            self.done = True
            return self.get_state(), 0, True

        return self.get_state(), 0, False

