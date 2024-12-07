import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist


# Hàm tính toán tổn thất PPO với clipping
def ppo_loss_actor(advantage, new_log_probs=None, old_log_probs=None,
                   sigma=None, entropy_weight=0.01, epsilon=0.2):
    """
    Hàm kết hợp tính loss cho actor với clipping và entropy.

    Args:
        advantage (Tensor): Advantage
        new_log_probs (Tensor): Log-probabilities mới (nếu có), shape (M,).
        old_log_probs (Tensor): Log-probabilities cũ (nếu có), shape (M,).
        pi_new (Tensor): Xác suất hành động theo policy mới (nếu có), shape (M,).
        pi_old (Tensor): Xác suất hành động theo policy cũ (nếu có), shape (M,).
        sigma_sq (Tensor): Phương sai
        entropy_weight (float): Trọng số cho entropy.
        epsilon (float): Giá trị clip, mặc định là 0.2.
        sigma : giá trị để tính entropy

    Returns:
        loss (Tensor): Giá trị loss trung bình.
    """
    if new_log_probs is not None and old_log_probs is not None:
        # Tính ratio
        ratios = torch.exp(new_log_probs - old_log_probs)
    else:
        raise ValueError("Phải cung cấp (new_log_probs, old_log_probs)")

    # Clipping ratio
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)

    # Tính surrogate loss
    surrogate_loss = torch.min(ratios * advantage, clipped_ratios * advantage)

    # Tính entropy
    if sigma is not None:
        sigma = torch.clamp(sigma, min=1e-3)  # Kiểm soát sigma ko được âm
        entropy = 0.5 * torch.sum(torch.log(2 * torch.pi * torch.e * sigma), dim=1)
    else:
        entropy = torch.zeros_like(surrogate_loss)

    # Tổng hợp loss
    loss = -torch.mean(surrogate_loss) + entropy_weight * torch.mean(entropy)
    return loss

def ppo_loss_critic(predicted_values, returns):
    """
    Loss của Critic trong PPO (mean squared error).

    Args:
        predicted_values: Giá trị state-value dự đoán (shape: [batch, time, 1]).
        returns: Giá trị kỳ vọng (shape: [batch, time, 1]).

    Returns:
        loss: Critic loss đã trung bình.
    """
    critic_loss = F.mse_loss(predicted_values, returns, reduction='mean')
    return critic_loss


def compute_log_pi(actions, mu, sigma):
    """
    Tính log-probability của các hành động dựa trên phân phối chuẩn.

    Args:
        actions: Tensor các hành động a (batch_size x action_dim)
        mu: Tensor giá trị trung bình mu
        sigma: Tensor độ lệch chuẩn sigma

    Returns:
        log_pi: Tensor log-probability
    """
    normal_dist = dist.Normal(mu, sigma)  # Tạo phân phối chuẩn với mu và sigma
    log_prob = normal_dist.log_prob(actions)  # Tính log-probability
    return log_prob.sum(dim=-1)  # Tổng log-probability trên tất cả các hành động


# Mô hình Actor sử dụng LSTM
class Actor(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        self.input_size = input_size
        self.output_size = output_size
        super(Actor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)  # Lớp LSTM đầu tiên
        self.norm1 = nn.LayerNorm(128)  # Chuẩn hóa layer để tăng ổn định
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout để tránh overfitting
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)  # Lớp LSTM thứ hai
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(128, 128)  # Giảm chiều đầu ra nhưng vì đầu ra là 80 nên không cần giảm
        self.output_layer = nn.Linear(128, output_size)  # Dự đoán giá trị đầu ra cuối cùng

    def forward(self, x):
        # Kiểm tra và chuyển `x` về thiết bị của mô hình
        device = next(self.parameters()).device  # Lấy thiết bị của mô hình
        if x.device != device:
            x = x.to(device)

        x, _ = self.lstm1(x)  # Đầu vào qua LSTM1
        x = self.norm1(x)  # Chuẩn hóa
        x = self.dropout1(x)  # Dropout
        x, _ = self.lstm2(x)  # Đầu vào qua LSTM2
        x = self.norm2(x)
        x = self.dropout2(x)
        # Qua projection
        x = self.projection(x)

        # Đầu ra cuối cùng
        output = self.output_layer(x)

        # Chia thành 2 phần: 30 giá trị đầu (mu), 30 giá trị cuối (sigma)
        mu = output[..., :30]  # Lấy 30 giá trị đầu tiên
        sigma = output[..., 30:]  # Lấy 30 giá trị cuối cùng

        # Đảm bảo sigma > 0 bằng cách sử dụng hàm softplus
        sigma = nn.functional.softplus(sigma)

        return mu, sigma

    # Lưu trạng thái mô hình
    def save_model(self, path):
        # Kiểm tra nếu file tồn tại
        if os.path.exists(path):
            os.remove(path)  # Xóa file cũ
            # print(f"File {path} existed and was deleted.")

        # Lưu mô hình vào file
        torch.save(self.state_dict(), path)
        # tạo thêm file cho việc hiển thị để tránh tranh chấp tài nguyên
        if 'actor' in str(path):
            # Tìm vị trí cần chèn
            insert_position = path.find("actor")
            # Tạo chuỗi mới bằng cách chèn "viewer_" vào
            if insert_position != -1:
                modified_path = path[:insert_position] + "viewer_" + path[insert_position:]
                torch.save(self.state_dict(), modified_path)
            else:
                modified_path = path  # Giữ nguyên nếu "actor_epoch" không tồn tại
                torch.save(self.state_dict(), modified_path)
        # print(f"Model saved successfully at {path}")

    # Tải trạng thái mô hình
    def load_model(self, path):
        with open(path, 'rb') as f:  # Đảm bảo file được đóng sau khi load
            model_state = torch.load(f, weights_only=True)
        self.load_state_dict(model_state)
        self.eval()  # tắt dropout và batch normalization


# Mô hình Critic sử dụng LSTM
class Critic(nn.Module):
    def __init__(self, input_size, dropout_rate=0.2):
        super(Critic, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.norm1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)  # Critic chỉ có 1 giá trị đầu ra (hàm giá trị)

    def forward(self, x):
        # Kiểm tra và chuyển `x` về thiết bị của mô hình
        device = next(self.parameters()).device  # Lấy thiết bị của mô hình
        if x.device != device:
            x = x.to(device)

        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.projection(x))
        return self.output_layer(x)  # Giá trị dự đoán cuối cùng

    # Lưu trạng thái mô hình
    def save_model(self, path):
        # Kiểm tra nếu file tồn tại
        if os.path.exists(path):
            os.remove(path)  # Xóa file cũ
            # print(f"File {path} existed and was deleted.")

        # Lưu mô hình vào file
        torch.save(self.state_dict(), path)
        # print(f"Model saved successfully at {path}")

    # Tải trạng thái mô hình
    def load_model(self, path):
        with open(path, 'rb') as f:  # Đảm bảo file được đóng sau khi load
            model_state = torch.load(f, weights_only=True)
        self.load_state_dict(model_state)
        self.eval()  # tắt dropout và batch normalization


class PPOClip_Training:
    def __init__(self,
                 training_id,
                 actor_model,
                 critic_model,
                 states,
                 actions,
                 log_probs,
                 sigma,
                 rewards,
                 returns,
                 advantages,
                 epsilon=0.2,
                 entropy_weight=0.01,
                 num_epochs=4,
                 learning_rate=0.0001,
                 path_dir="models/param/"
                 ):
        self.training_id = training_id
        self.path_dir = path_dir
        self.actor = actor_model
        self.critic = critic_model
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.sigma = sigma
        self.rewards = rewards
        self.returns = returns
        self.advantages = advantages
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.epochs = num_epochs

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

    """
    Hàm huấn luyện chính cho PPO với clipping.

    Args:
        :param actor_model: Mô hình policy actor
        :param critic_model: Mô hình ước lượng Value critic
        :param states: Tensor trạng thái (shape: [batch, time, state_dim]).
        :param actions: Tensor hành động (shape: [batch, time, action_dim]).
        :param log_probs: Log-probabilities cũ (shape: [batch, time, 1]).
        :param epsilon: Hệ số clipping.
        :param advantages: Giá trị advantage (shape: [batch, time, 1]).
        :param returns: Giá trị kỳ vọng (shape: [batch, time, 1]).
        :param num_epochs: Số epochs
        :param batch_size: Số lượng batch mỗi lần huấn luyện
        :param learning_rate: Tốc độ học

    Returns:
        None

    """
    # Biến lớp, dùng chung cho tất cả các đối tượng
    best_reward = -float('inf')  # Đây là biến lớp

    def train(self):
        # Tạo các danh sách lưu trữ giá trị
        epochs_history = []
        rewards_history = []
        entropy_history = []
        actor_loss_history = []
        critic_loss_history = []
        # Chuẩn hóa advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        for epoch in range(self.epochs):
            # ** ------------- Huấn luyện Actor -----------------**
            # Forward qua model để tính các giá trị mới
            mu, sigma = self.actor(self.states)  # Tương thích với cấu trúc batch

            # Tính log-probabilities mới
            new_log_probs = compute_log_pi(self.actions, mu, sigma).unsqueeze(-1)  # Thêm chiều để khớp shape

            # Tính loss
            actor_loss = ppo_loss_actor(advantage=self.advantages,
                                        new_log_probs=new_log_probs,
                                        old_log_probs=self.log_probs,
                                        sigma=self.sigma,
                                        entropy_weight=self.entropy_weight,
                                        epsilon=self.epsilon)
            # Backward Actor
            self.actor_optimizer.zero_grad()
            # actor_loss.backward()  # Tính gradient cho Actor
            actor_loss.backward()  # Tính backward để giữ đồ thị
            # # In giá trị gradient của các tham số trong Actor
            # print("\n=== Gradient của Actor ===")
            # for name, param in self.actor.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm(2).item()  # Tính L2 norm của gradient
            #         print(f"{name} - Gradient norm: {grad_norm}")
            #     else:
            #         print(f"{name} không có gradient!")
            self.actor_optimizer.step()  # Cập nhật tham số của Actor

            # ** ------------- Huấn luyện Critic -----------------*
            # Backward Critic
            # Forward qua model để tính các giá trị mới
            predict_critic = self.critic(self.states)
            # Tính loss
            critic_loss = ppo_loss_critic(predict_critic, self.returns)
            self.critic_optimizer.zero_grad()
            # self.critic_loss.backward()  # Tính gradient cho Critic
            critic_loss.backward()  # Tính backward để giữ đồ thị
            # In giá trị gradient của các tham số trong Critic
            # print("\n=== Gradient của Critic ===")
            # for name, param in self.critic.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm(2).item()  # Tính L2 norm của gradient
            #         print(f"{name} - Gradient norm: {grad_norm}")
            #     else:
            #         print(f"{name} không có gradient!")

            self.critic_optimizer.step()  # Cập nhật tham số của Critic

            print(
                f"Epoch {epoch + 1}/{self.epochs}: Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

            # Lưu mô hình sau khi xong 1 iterator (4 epoch)
            # Tính entropy để theo dõi mức độ ngẫu nhiên của policy
            epochs_history.append(self.epochs * self.training_id + epoch + 1)
            entropy = -torch.sum(sigma.log(), dim=-1).mean()  # Entropy tính từ sigma
            entropy_history.append(entropy.item())  # Lưu lại Entropy
            actor_loss_history.append(actor_loss.item())  # Lưu lại Actor loss
            critic_loss_history.append(critic_loss.item())  # Lưu lại Critic loss
            # Tính reward trung bình
            mean_reward = self.returns.mean().item()
            rewards_history.append(mean_reward)  # Lưu lại reward

            # =========== LƯU THÔNG TIN MÔ HÌNH SAU KHI HUẤN LUYỆN ===================

            # So sánh phần thưởng tốt nhất và lưu mô hình nếu có cải thiện
            if mean_reward > PPOClip_Training.best_reward:
                PPOClip_Training.best_reward = mean_reward
                best_actor_model = self.actor.state_dict()  # Lưu tham số của actor model
                best_critic_model = self.critic.state_dict()  # Lưu tham số của critic model

                # Lưu mô hình tốt nhất
                torch.save({
                    'actor_model': best_actor_model,
                    'critic_model': best_critic_model,
                    'reward': PPOClip_Training.best_reward,
                }, f"{self.path_dir}best_model.pth")

            # # Sau mỗi epoch, lưu mô hình Actor và Critic
            self.actor.save_model(f"{self.path_dir}actor_epoch_latest.pth")
            self.critic.save_model(f"{self.path_dir}critic_epoch_latest.pth")

            # # Sau mỗi epoch, lưu mô hình Actor và Critic cuối cùng
            # num_epoch = self.epochs * self.training_id + epoch + 1
            # self.actor.save_model(f"{self.path_dir}actor_epoch_{num_epoch}.pth")
            # self.critic.save_model(f"{self.path_dir}critic_epoch_{num_epoch}.pth")

        # Kiểm tra file log đã tồn tại hay chưa
        log_file = f"{self.path_dir}training_metrics_all.pth"
        if os.path.exists(log_file):
            # Nếu đã có log, load dữ liệu cũ
            training_log = torch.load(log_file, weights_only=True)
        else:
            # Nếu chưa có log, tạo file mới
            training_log = []

        # Thêm dữ liệu mới vào log (chỉ lấy dữ liệu epoch cuối)
        training_log.append({
            "epochs_history": epochs_history[-1],
            "rewards_history": rewards_history[-1],
            "entropy_history": entropy_history[-1],
            "actor_loss_history": actor_loss_history[-1],
            "critic_loss_history": critic_loss_history[-1]
        })

        # Lưu lại toàn bộ log
        torch.save(training_log, log_file)

# ==================== TÌM THAM SỐ MÔ HÌNH VÀ LOAD LÊN MODLE =================


def find_latest_model(prefix, directory="."):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pth")]

    if not files:
        return None

    # Sắp xếp file theo thời gian sửa đổi gần nhất
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)

    # Trả về file mới nhất
    return os.path.join(directory, files[0])


# def find_latest_model(prefix, directory="."):
#     """
#     Tìm file model mới nhất trong thư mục với prefix.
#
#     Args:
#         prefix (str): Tiền tố của file (vd: 'actor_epoch_').
#         directory (str): Thư mục chứa các file model.
#
#     Returns:
#         str: Đường dẫn đến file model mới nhất.
#     """
#     files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pth")]
#     if not files:
#         return None  # Không tìm thấy file
#     # Sắp xếp file theo số epoch
#     files = sorted(files, key=lambda x: int(re.search(r'\d+', x).group()), reverse=True)
#     return os.path.join(directory, files[0])  # File mới nhất

