import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
import torch
from spicy import ndimage
import random

class CuttingGlassEnv(gym.Env):
    """Môi trường Gym cho bài toán cắt kính từ phôi kính"""

    def __init__(self, num_stocks=10, num_products=10):
        super(CuttingGlassEnv, self).__init__()

        # Số lượng phôi kính và sản phẩm
        self.num_stocks = num_stocks
        self.num_products = num_products

        # Khởi tạo danh sách phôi kính và sản phẩm
        self.stocks = self.generate_stocks()
        self.products = self.generate_products()

        # Kích thước lớn nhất của phôi kính
        self.max_stock_size = (100, 100)

        # Không gian hành động: (stock_idx, x, y, rotate)
        self.action_space = spaces.MultiDiscrete([num_stocks, 100, 100, 2])

        # Không gian quan sát
        self.observation_space = spaces.Box(low=-1, high=1, shape=(100, 100), dtype=np.float32)

        self.current_stock_idx = 0
        self.current_product = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_stocks(self):
        """Tạo danh sách phôi kính từ một tập hợp cố định"""
        stock_sizes = [(100, 100), (90, 90), (80, 80), (70, 70), (60, 60)]
        stocks = []
        for _ in range(self.num_stocks):
            w, h = random.choice(stock_sizes)
            stock = np.zeros((w, h), dtype=int)  # 0: vùng khả dụng
            stocks.append(stock)
        return stocks

    def generate_products(self):
        """Tạo danh sách sản phẩm từ một tập hợp cố định"""
        product_sizes = [(30, 30), (25, 25), (20, 20), (15, 15), (10, 10)]
        products = []
        for _ in range(self.num_products):
            w, h = random.choice(product_sizes)
            products.append((w, h))
        return products

    def pad_stock(self, stock):
        """Đảm bảo phôi luôn có kích thước cố định 100x100"""
        if isinstance(stock, torch.Tensor):
            stock = stock.to(self.device) # Chuyển Tensor -> Numpy nếu cần

        if len(stock.shape) == 3:  # Nếu có batch size, loại bỏ nó
            stock = stock.squeeze(0)

        if len(stock.shape) != 2:
            raise ValueError(f"Expected 2D array, but got shape {stock.shape}")

        max_width, max_height = 100, 100
        padded_stock = np.full((max_width, max_height), fill_value=-1, dtype=int)
        w, h = stock.shape
        padded_stock[:w, :h] = stock
        return padded_stock



    def reset(self, seed=None):
        """Đặt lại môi trường"""
        np.random.seed(seed)
        self.stocks = self.generate_stocks()
        self.products = self.generate_products()
        self.current_stock_idx = 0
        self.current_product = 0
        return self.pad_stock(self.stocks[self.current_stock_idx])
    
    def can_place_product(self, stock, w, h):
        """Kiểm tra xem có thể đặt sản phẩm trên phôi không"""
        stock_w, stock_h = stock.shape
        for x in range(stock_w - w + 1):
            for y in range(stock_h - h + 1):
                if np.all(stock[x:x+w, y:y+h] == 0):  # Kiểm tra vùng khả dụng
                    return True
        return False
    

    
    def calculate_reward(self, stock, previous_stock_state):
        """Tính reward dựa trên các tiêu chí tối ưu hóa"""
        # Diện tích ban đầu của stock
        initial_area = stock.shape[0] * stock.shape[1]
        
        # Tính diện tích đã sử dụng
        used_area = np.sum(previous_stock_state == -1)  # Vùng đã cắt
        current_used_area = np.sum(stock == -1)

        # Tính diện tích thừa (phần chưa sử dụng)
        previous_waste_ratio = (initial_area - used_area) / initial_area
        current_waste_ratio = (initial_area - current_used_area) / initial_area

        # Reward dựa trên việc giảm phần thừa
        waste_reward = previous_waste_ratio - current_waste_ratio  # Nếu diện tích thừa giảm -> reward dương

        # Reward dựa trên số lượng stock đã dùng
        stock_usage_reward = -0.1 * self.current_stock_idx  # Sử dụng ít stock hơn thì tốt

        # Tính chu vi tổng thể của các sản phẩm đã cắt
        def calculate_total_perimeter(stock):
            labeled, num_features = ndimage.label(stock == -1)  # Nhãn các vùng đã cắt
            perimeter = 0
            for i in range(1, num_features + 1):
                region = (labeled == i)
                perimeter += np.sum(region) - np.sum(ndimage.binary_erosion(region))  # Chu vi = điểm biên
            return perimeter

        previous_perimeter = calculate_total_perimeter(previous_stock_state)
        current_perimeter = calculate_total_perimeter(stock)
        perimeter_reward = previous_perimeter - current_perimeter  # Nếu chu vi giảm, reward dương

        # Tổng reward
        total_reward = (waste_reward * 5) + (stock_usage_reward * 2) + (perimeter_reward * 3)

        return np.clip(total_reward, -20, 20)


    def step(self, action):
        """Thực hiện hành động """
        stock_idx, x, y, rotate = action

        # Kiểm tra stock có hợp lệ không
        if stock_idx < 0 or stock_idx >= self.num_stocks:
            return self.pad_stock(self.stocks[self.current_stock_idx]), -5, False, {}  # Phạt nặng hơn nếu chọn stock không hợp lệ

        stock = self.stocks[stock_idx]  # Lấy phôi mà agent chọn
        w, h = self.products[self.current_product]  # Lấy kích thước sản phẩm hiện tại

        if rotate == 1:  # Nếu quay sản phẩm, đổi chiều rộng và cao
            w, h = h, w

        # Kiểm tra xem có thể đặt sản phẩm vào stock này không
        if not self.can_place_product(stock, w, h):
            # Nếu không thể đặt, phạt nặng hơn để mô hình học cách chọn stock phù hợp
            penalty = -2  
            
            # Thử tất cả các stock còn lại xem có cái nào phù hợp không
            valid_stock_found = False
            for i in range(self.num_stocks):
                if i != stock_idx and self.can_place_product(self.stocks[i], w, h):
                    self.current_stock_idx = i  # Chuyển sang stock phù hợp
                    valid_stock_found = True
                    break
            
            # Nếu không tìm được stock nào phù hợp, kết thúc episode
            if not valid_stock_found:
                return self.pad_stock(stock), -10, True, {}  # Phạt rất nặng và kết thúc episode

            return self.pad_stock(self.stocks[self.current_stock_idx]), penalty, False, {}

        # Kiểm tra vị trí cắt có hợp lệ không
        if x + w > stock.shape[0] or y + h > stock.shape[1] or np.any(stock[x:x+w, y:y+h] != 0):
            return self.pad_stock(stock), -3, False, {}  # Phạt nếu chọn vị trí không hợp lệ

        previous_stock_state = np.copy(self.stocks[self.current_stock_idx])  # Lưu trạng thái trước đó

        # Nếu hợp lệ, tiến hành cắt sản phẩm từ stock
        self.stocks[self.current_stock_idx][x:x+w, y:y+h] = -1

        # Tính reward dựa trên trạng thái trước và sau khi cắt
        reward = self.calculate_reward(self.stocks[self.current_stock_idx], previous_stock_state)

        self.current_product += 1  # Chuyển sang sản phẩm tiếp theo

        # Kiểm tra điều kiện kết thúc episode
        done = self.current_product >= self.num_products

        return self.pad_stock(self.stocks[self.current_stock_idx]), reward, done, {}



    def render(self):
        """Hiển thị trạng thái phôi kính"""
        plt.imshow(self.stocks[self.current_stock_idx], cmap="gray")
        plt.title(f"Cutting Glass - Stock {self.current_stock_idx + 1}")
        plt.show()
