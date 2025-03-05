import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOPolicy(nn.Module):
    def __init__(self, num_stocks, learning_rate=0.0003):
        super(PPOPolicy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # mạng CNN để trích xuất đặc trưng, đầu vào là (1,100,100) , đầu ra là (16,5,5): 400 vector đặc trưng
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),            
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((5, 5)),  # Giảm feature map xuống (5,5)
            nn.Flatten()
        )
        # Actor: Nhận feature map từ CNN, đầu ra là phân phối xác suất của hành động
        fc_input_size = 16 * 5 * 5
        self.actor_fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),      # mạng fully connected với 128 units và ReLU activation
            nn.ReLU(),
            nn.Linear(128, num_stocks + 100 + 100 + 2)  # Output size phù hợp với action space (10 + 100 + 100 + 2)
        )

        # Critic: Nhận feature map từ CNN, đầu ra là 1 giá trị ước lượng (ước lượng giá trị kỳ vọng của trạng thái) để cập nhật chính sách
        self.critic_fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        features = self.cnn(state)
        logits = self.actor_fc(features)
        logits = torch.log_softmax(logits, dim=-1)  # Dùng log softmax để tránh lỗi Nan khi có số quá nhỏ hoặc quá lớn

        # Dùng exp để chuyển logits sang xác suất hợp lệ
        stock_dist = Categorical(probs=logits[:, :10])    # 10 phần tử đầu tiên là phân phối xác suất của stock
        x_dist = Categorical(probs=logits[:, 10:110])    # 100 phần tử tiếp theo là phân phối xác suất của x
        y_dist = Categorical(probs=logits[:, 110:210])    # 100 phần tử tiếp theo là phân phối xác suất của y
        rotate_dist = Categorical(probs=logits[:, -2:])   # 2 phần tử cuối cùng là phân phối xác suất của rotate

        value = self.critic_fc(features)            # Dùng critic để ước lượng giá trị trạng thái từ đặc trưng
        return stock_dist, x_dist, y_dist, rotate_dist, value       # trả về phân phối xác suất và giá trị ước lượng (policy và value function)
