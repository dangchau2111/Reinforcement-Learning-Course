# Proximal Policy Optimization (PPO) 
Thuật toán Proximal Policy Optimization (PPO) là một trong những phương pháp học tăng cường (Reinforcement Learning) dựa trên gradient chính sách, được phát triển nhằm đạt được sự ổn định và hiệu quả trong quá trình huấn luyện. Dưới đây là một số điểm chính giới thiệu về PPO:
- Cơ chế cơ bản:\
  PPO thuộc nhóm các thuật toán actor-critic, trong đó có hai thành phần chính:
  * Actor (Chính sách): Sinh ra hành động dựa trên trạng thái hiện tại của môi trường.
  * Critic (Giá trị): Ước lượng giá trị của trạng thái (state value), giúp đánh giá hiệu suất của chính sách.
    ![image](https://github.com/user-attachments/assets/9ed646e0-7b12-4dd4-bea8-eb904c315bcd)

- Mục tiêu huấn luyện:
  * PPO tối ưu hóa một hàm mục tiêu gọi là surrogate objective. Hàm mục tiêu này đo lường hiệu quả của việc cập nhật chính sách mới so với chính sách cũ và sử dụng một cơ chế clipping để giới hạn sự thay đổi quá lớn của chính sách. Điều này giúp đảm bảo quá trình huấn luyện không bị mất ổn định do những bước cập nhật quá mạnh.
  
$$
\nabla_\theta L^{\mathrm{CLIP}}(\theta) 
= \nabla_\theta \min \Big( 
    r_t(\theta)\,A_t,\; 
    \mathrm{clip}\big(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon\big)\,A_t 
\Big)
$$

- Ưu điểm nổi bật:
  * Đơn giản và hiệu quả: PPO được đánh giá cao về mặt tính đơn giản khi triển khai và khả năng đạt hiệu quả cao trong nhiều bài toán RL.
  * Ổn định trong huấn luyện: Cơ chế clipping của PPO giúp tránh việc cập nhật chính sách quá mạnh, từ đó đảm bảo quá trình học ổn định hơn so với một số thuật toán policy gradient truyền thống.
  * Tính ứng dụng rộng rãi: PPO đã được áp dụng thành công trong nhiều lĩnh vực như game, robot, và các bài toán tối ưu hoá phức tạp khác.
- Nhược điểm:
  * Độ nhạy với hyperparameter: PPO có thể khá nhạy cảm với việc lựa chọn các siêu tham số như learning rate, hệ số clipping, số lượng epoch huấn luyện và kích thước batch. Việc thiết lập không phù hợp có thể dẫn đến quá trình huấn luyện không ổn định hoặc hiệu suất kém.
  * Khó khăn trong không gian hành động phức tạp: Mặc dù PPO hoạt động tốt trên nhiều bài toán, nhưng trong một số trường hợp có không gian hành động liên tục hoặc rất lớn, việc thiết kế mạng và cấu trúc phân phối có thể trở nên phức tạp và đòi hỏi phải tùy chỉnh cẩn thận.
- Quy trình cập nhật:\
Trong quá trình huấn luyện, PPO thu thập các trải nghiệm (experience) từ môi trường và tính toán các giá trị advantages (lợi thế) dựa trên sự chênh lệch giữa giá trị thực tế (return) và giá trị dự đoán (value). Sau đó, thuật toán cập nhật chính sách dựa trên hàm mục tiêu đã được clipping, nhằm giới hạn sự khác biệt giữa chính sách cũ và mới, đảm bảo quá trình huấn luyện được ổn định.

# Áp dụng PPO để giải quyết bài toán Cutting Stock.
Trong repo này của tôi: 
  - File cutting_glass_env.py định nghĩa một môi trường mô phỏng để cắt kính (CuttingGlass) dựa trên OpenAI Gym:
    * Khởi tạo và xử lý dữ liệu.
    * Tạo valid mask.
    * Xử lý hành động (step).
    * Tính toán reward.
    * Reset môi trường.
  - File ppo_policy.py:
    * Định nghĩa mạng CNN, MLP để trích xuất đặc trưng, một mô hình Actor-Critic dùng cho thuật toán PPO.
    * Định nghĩa luồng forward để xử lý dữ liệu đầu vào, trả về hành động, phân phối xác suất hành động và giá trị trạng thái V(s).
  - File ppo_agent.py:
    * Định nghĩa một lớp PPOAgent.
    * Triển khai quá trình lưu trữ trải nghiệm.
    * Chọn hành động và huấn luyện policy dựa trên thuật toán PPO.
  - File main.py là một script huấn luyện cho agent:
    * Huấn luyện và lưu mô hình vào folder Model.
    * Lưu lại plot về loss và reward của mô hình ở folder Loss_Plot.
  - File visualize.py:
    * Kiểm tra hiệu suất của model trên 10 tập data khác nhau.
  - File test.py:
    * Visualize kết quả cắt của model trên 1 tập data.
  ![image](https://github.com/user-attachments/assets/24b067f2-4720-4094-9b0c-75d4fcf72e9d)


Nguồn tham khảo: \
https://medium.com/%40oleglatypov/a-comprehensive-guide-to-proximal-policy-optimization-ppo-in-ai-82edab5db200   \
https://arxiv.org/pdf/1707.06347   \
https://www.youtube.com/watch?v=hlv79rcHws0   

