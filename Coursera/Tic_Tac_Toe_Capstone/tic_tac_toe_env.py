import numpy as np

class TicTacToeEnv:
    def __init__(self):
        """Khởi tạo môi trường Tic-Tac-Toe"""
        self.state = np.zeros((3, 3), dtype=int)  # Bảng Tic Tac Toe (3x3)
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
        return tuple(self.state.flatten())  # Chuyển ma trận thành vector 1 chiều

    def check_winner(self):
        """Kiểm tra điều kiện thắng"""
        for i in range(3):
            # Kiểm tra hàng ngang
            if np.all(self.state[i, :] == 1):
                return 1  # Bot (X) thắng
            if np.all(self.state[i, :] == 2):
                return 2  # Người chơi (O) thắng

            # Kiểm tra cột dọc
            if np.all(self.state[:, i] == 1):
                return 1
            if np.all(self.state[:, i] == 2):
                return 2

        # Kiểm tra đường chéo chính
        if np.all(np.diag(self.state) == 1):
            return 1
        if np.all(np.diag(self.state) == 2):
            return 2

        # Kiểm tra đường chéo phụ
        if np.all(np.diag(np.fliplr(self.state)) == 1):
            return 1
        if np.all(np.diag(np.fliplr(self.state)) == 2):
            return 2

        return None  # Không ai thắng

    def step(self, action, player):
        """Thực hiện một bước đi"""
        if self.state[action // 3, action % 3] != 0 or self.done:
            return self.get_state(), -10, True  # Hành động không hợp lệ (trừ điểm)

        # Gán giá trị vào bảng cờ (1 = bot, 2 = người chơi)
        self.state[action // 3, action % 3] = player

        # Kiểm tra xem có ai thắng không
        winner = self.check_winner()

        if winner is not None:
            self.done = True
            self.winner = winner
            return self.get_state(), 5 if winner == 1 else -5, True  # Phần thưởng khi thắng, thua

        # Kiểm tra hòa (không còn ô trống)
        if not (self.state == 0).any():
            self.done = True
            return self.get_state(), 0, True  # Hòa, không có phần thưởng

        return self.get_state(), 0, False  # Trò chơi tiếp tục
