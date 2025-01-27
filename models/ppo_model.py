# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def create_mask(batch_size, timestamps, lengths_traj, device):
    mask = torch.zeros((batch_size, timestamps), device=device)
    for i, length in enumerate(lengths_traj):
        mask[i, :length] = 1
    return mask


# Hàm tính toán tổn thất PPO với clipping
def ppo_loss_actor(advantage, new_log_probs=None, old_log_probs=None, sigma=None, entropy_weight=0.01, epsilon=0.2,
                   mask=None):
    if mask is not None:
        # Mở rộng mask để phù hợp với advantage
        mask_expanded = mask.unsqueeze(-1)  # Kích thước [batch_size, trajectory_length, 1]
    else:
        raise ValueError("Phải cung cấp mask")

    if new_log_probs is not None and old_log_probs is not None:
        ratios = torch.exp(
            torch.sum((new_log_probs - old_log_probs) * mask_expanded, dim=2, keepdim=True))  # dim=2 loại action_dim
    else:
        raise ValueError("Phải cung cấp (new_log_probs, old_log_probs)")

    # ** Chuẩn hóa advantage **

    # Tính tổng có trọng số
    advantage_sum = (advantage * mask_expanded).sum(dim=1, keepdim=True)  # [batch_size, 1, 1]

    # Đếm số phần tử hợp lệ
    valid_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # [batch_size, 1, 1]

    # Tính Mean
    advantage_mean = advantage_sum / (valid_count + 1e-8)  # [batch_size, 1, 1]

    # Tính phương sai (Variance)
    advantage_squared_diff = (advantage - advantage_mean) ** 2
    weighted_variance = (advantage_squared_diff * mask_expanded).sum(dim=1, keepdim=True) / (valid_count + 1e-8)

    # Tính Std (Căn bậc hai của Variance)
    advantage_std = torch.sqrt(weighted_variance)  # [batch_size, 1, 1]

    # Chuẩn hóa advantage
    advantage_norm = (advantage - advantage_mean) / (advantage_std + 1e-8)  # [batch_size, trajectory_length, 1]

    # Tổng hợp các hành động lại
    clipped_ratios = torch.clamp(ratios * advantage_norm, (1 - epsilon) * advantage_norm, (1 + epsilon) * advantage_norm)

    # Tính surrogate loss
    surrogate_loss = torch.min(ratios * advantage_norm, clipped_ratios)
    surrogate_loss = surrogate_loss * mask_expanded

    # # Tính entropy
    # if sigma is not None:
    #     entropy = 0.5 * torch.sum(torch.log(2 * torch.pi * torch.e * sigma ** 2) * mask_expanded, dim=2, keepdim=True)
    # else:
    #     entropy = torch.zeros_like(surrogate_loss)
    # print(f"ratios: {ratios}")
    # print(f"clipped_ratios: {clipped_ratios}")
    # print(f"sigma: {sigma}")
    # print(f"surrogate_loss: {surrogate_loss}")
    # Tổng hợp loss
    # _loss = surrogate_loss + entropy_weight * entropy
    _loss = surrogate_loss 
    loss_sum = (_loss * mask_expanded).sum(dim=1, keepdim=True)  # [batch_size, 1, 1]

    # Đếm số phần tử hợp lệ
    valid_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # [batch_size, 1, 1]

    # Tính Mean loss của trajectory
    loss_traj_mean = loss_sum / (valid_count + 1e-8)  # [batch_size, 1, 1]

    # Tính mean loss của batch
    return loss_traj_mean.mean()


def ppo_loss_critic(predicted_values, returns, mask=None):
    if mask is not None:
        # Mở rộng mask để phù hợp với advantage
        mask_expanded = mask.unsqueeze(-1)  # Kích thước [batch_size, trajectory_length, 1]
    else:
        raise ValueError("Phải cung cấp mask")
    # Đếm số phần tử hợp lệ
    valid_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # [batch_size, 1, 1]
    # ** Chuẩn hóa advantage **

    # # Tính tổng có trọng số
    # returns_sum = (returns * mask_expanded).sum(dim=1, keepdim=True)  # [batch_size, 1, 1]

    # # Tính Mean
    # returns_mean = returns_sum / (valid_count + 1e-8)  # [batch_size, 1, 1]
    #
    # # Tính phương sai (Variance)
    # returns_squared_diff = (returns - returns_mean) ** 2
    # weighted_variance = (returns_squared_diff * mask_expanded).sum(dim=1, keepdim=True) / (valid_count + 1e-8)
    #
    # # Tính Std (Căn bậc hai của Variance)
    # returns_std = torch.sqrt(weighted_variance)  # [batch_size, 1, 1]
    #
    # # Chuẩn hóa advantage
    # returns_norm = (returns - returns_mean) / (returns_std + 1e-8)  # [batch_size, trajectory_length, 1]
    # Chỉ tính toán trên các giá trị không phải padding
    # Tính hiệu giữa giá trị dự đoán và giá trị thực
    differences = predicted_values - returns
    # Tính bình phương của các hiệu
    squared_differences = differences ** 2
    # Tính trung bình của các bình phương hiệu
    loss_sum = (squared_differences * mask_expanded).sum(dim=1, keepdim=True)  # [batch_size, 1, 1]
    # Tính Mean loss của trajectory
    loss_traj_mean = loss_sum / (valid_count + 1e-8)  # [batch_size, 1, 1]
    return loss_traj_mean.mean()


def compute_log_pi(actions, mu, sigma, ):
    normal_dist = dist.Normal(mu, sigma)  # Tạo phân phối chuẩn với mu và sigma
    log_prob = normal_dist.log_prob(actions)  # Tính log-probability

    return log_prob


# Mô hình Actor sử dụng LSTM
class Actor(nn.Module):
    def __init__(self, pTarget_range, input_size=None, output_size=None,
                 lengths_traj=None, sigma_warmUp=None,
                 dropout_rate=0.2):
        self.input_size = input_size
        self.output_size = output_size
        self.sigma_warmUp = sigma_warmUp
        # self.dynamics_randomization = dynamics_randomization
        self.pTarget_min = torch.tensor([v[0] for v in pTarget_range.values()])
        self.pTarget_max = torch.tensor([v[1] for v in pTarget_range.values()])

        self.lengths_traj = lengths_traj  # Xử lý các dữ liệu padding khi train
        super(Actor, self).__init__()
        # LSTM layers
        self.lstm1 = nn.LSTM(self.input_size, 128, batch_first=True)  # Lớp LSTM đầu tiên
        # self.activation1 = nn.ReLU()  # Hàm kích hoạt sau LSTM1
        # self.dropout1 = nn.Dropout(dropout_rate)  # Dropout 1
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)  # Lớp LSTM thứ hai
        # self.activation2 = nn.ReLU()  # Hàm kích hoạt sau LSTM1
        # self.dropout2 = nn.Dropout(dropout_rate)  # Dropout 2

        # Fully Connected layer
        self.output_layer = nn.Sequential(
            nn.Linear(128, self.output_size),  # Kết nối đầy đủ
            # nn.Dropout(dropout_rate)  # Dropout cuối
        )

    def forward(self, x):
        # Kiểm tra và chuyển `x` về thiết bị của mô hình
        device = next(self.parameters()).device  # Lấy thiết bị của mô hình
        if x.device != device:
            x = x.to(device)

        if self.pTarget_min.device != device:
            self.pTarget_min = self.pTarget_min.to(device)
            self.pTarget_max = self.pTarget_max.to(device)

        # Xử lý padding khi train
        if self.lengths_traj is not None:
            x = pack_padded_sequence(x, self.lengths_traj, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm1(x)
        # if self.lengths_traj is not None:
        #     x, _ = pad_packed_sequence(x, batch_first=True)  # Giải nén PackedSequence về tensor
        # x = self.activation1(x)
        # x = self.dropout1(x)
        # if self.lengths_traj is not None:
        #     x = pack_padded_sequence(x, self.lengths_traj, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm2(x)
        # x = self.activation2(x)
        # x = self.dropout2(x)
        if self.lengths_traj is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)  # Giải nén PackedSequence về tensor
        # Fully Connected layer
        output = self.output_layer(x)
        # Đầu ra cuối cùng

        # Kiểm tra đầu ra của LSTM
        if self.lengths_traj is not None:
            mask = create_mask(output.size(0), output.size(1), self.lengths_traj, device)
        else:
            mask = None

        # Chia thành 2 phần: 30 giá trị đầu (mu), 30 giá trị cuối (sigma)
        mu = output[..., :30]  # Lấy 40 giá trị đầu tiên
        sigma = output[..., 30:]  # Lấy 40 giá trị cuối cùng

        # Tách thành 3 phần: pTarget_mu, pGain_mu, dGain_mu
        pTarget_mu = mu[..., :10]  # Lấy 10 giá trị đầu tiên từ mu
        pGain_mu = mu[..., 10:20]  # Lấy 10 giá trị tiếp theo từ mu
        dGain_mu = mu[..., 20:30]  # Lấy 10 giá trị cuối từ mu

        # Áp dụng tanh cho pTarget_mu (để chuẩn hóa trong khoảng [-1, 1])
        # print("MIN MAX: ", self.pTarget_max, self.pTarget_min)
        pTarget_mu = self.pTarget_min + (self.pTarget_max - self.pTarget_min) * (torch.tanh(pTarget_mu) + 1) / 2
        # print("ĐẦU RA: ", pTarget_mu)

        # Gộp các mu lại
        mu = torch.cat([pTarget_mu, pGain_mu, dGain_mu], dim=-1)

        # Đảm bảo sigma > 0 bằng cách sử dụng hàm softplus
        # sigma = nn.functional.softplus(sigma)
        # print(self.sigma_warmUp)
        # a, b = 0.01, 1
        # sigma = (b - a) * torch.sigmoid(output) + a
        # raise "TẮT"
        if self.sigma_warmUp is None:
            sigma = torch.clamp(nn.functional.softplus(sigma), min=0.01)
        else:
            sigma = torch.clamp(nn.functional.softplus(sigma), min=self.sigma_warmUp)

        # print("SHAPE sigma:", sigma.shape)

        if mask is None:
            mu = mu.reshape(-1)
            sigma = sigma.reshape(-1)
            return mu, sigma
        else:
            return mu, sigma, mask

    # Lưu trạng thái mô hình
    def save_model(self, path):

        try:
            torch.save(self.state_dict(), path)
            # print(f"Model saved successfully at {path}")
        except Exception as e:
            pass
            print(f"Error saving model to {path}: {e}")

        # tạo thêm file cho việc hiển thị để tránh tranh chấp tài nguyên
        if 'actor' in str(path):
            # Tìm vị trí cần chèn
            insert_position = path.find("actor")
            # Tạo chuỗi mới bằng cách chèn "viewer_" vào
            if insert_position != -1:
                modified_path = path[:insert_position] + "viewer_" + path[insert_position:]
                try:
                    torch.save(self.state_dict(), modified_path)
                    # print(f"Model saved successfully at {modified_path}")
                except Exception as e:
                    print(f"Error saving model to {modified_path}: {e}")
            else:
                modified_path = path  # Giữ nguyên nếu "actor_epoch" không tồn tại
                try:
                    torch.save(self.state_dict(), modified_path)
                    # print(f"Model saved successfully at {modified_path}")
                except Exception as e:
                    print(f"Error saving model to {modified_path}: {e}")
        # print(f"Model saved successfully at {path}")

    # Tải trạng thái mô hình
    def load_model(self, path):
        try:
            # Tải trọng số (weights_only=True để tránh cảnh báo)
            model_state = torch.load(path, weights_only=True)
            self.load_state_dict(model_state)
            self.eval()  # Đặt chế độ inference
            # print(f"Model weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")
            raise f"KHÔNG LOAD ĐƯỢC ACTOR"


class Critic(nn.Module):
    def __init__(self, input_size=None, lengths_traj=None, dropout_rate=0.1):
        self.input_size = input_size
        self.lengths_traj = lengths_traj  # Xử lý padding khi train
        super(Critic, self).__init__()
        # LSTM layers
        self.lstm1 = nn.LSTM(self.input_size, 128, batch_first=True)  # Lớp LSTM đầu tiên
        # self.activation1 = nn.ReLU()  # Hàm kích hoạt sau LSTM1
        # self.dropout1 = nn.Dropout(dropout_rate)  # Dropout 1
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)  # Lớp LSTM thứ hai
        # self.activation2 = nn.ReLU()  # Hàm kích hoạt sau LSTM1
        # self.dropout2 = nn.Dropout(dropout_rate)  # Dropout 2

        # Fully Connected layer
        self.output_layer = nn.Sequential(
            nn.Linear(128, 1),  # Kết nối đầy đủ
            # nn.Dropout(dropout_rate)  # Dropout cuối
        )

    def forward(self, x):
        # Kiểm tra và chuyển x về thiết bị của mô hình
        device = next(self.parameters()).device  # Lấy thiết bị của mô hình
        if x.device != device:
            x = x.to(device)

        # Xử lý padding khi train
        if self.lengths_traj is not None:
            x = pack_padded_sequence(x, self.lengths_traj.int(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm1(x)
        # if self.lengths_traj is not None:
        #     x, _ = pad_packed_sequence(x, batch_first=True)  # Giải nén PackedSequence về tensor
        # x = self.activation1(x)
        # x = self.dropout1(x)
        # if self.lengths_traj is not None:
        #     x = pack_padded_sequence(x, self.lengths_traj, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm2(x)
        # x = self.activation2(x)
        # x = self.dropout2(x)
        if self.lengths_traj is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)  # Giải nén PackedSequence về tensor
        # Fully Connected layer
        output = self.output_layer(x)

        return output  # Giá trị dự đoán cuối cùng

    # Lưu trạng thái mô hình
    def save_model(self, path):
        try:
            torch.save(self.state_dict(), path)
            # print(f"Model saved successfully at {path}")
        except Exception as e:
            print(f"Error saving model to {path}: {e}")

        # Tải trạng thái mô hình

    def load_model(self, path):
        try:
            # Tải trọng số (weights_only=True để tránh cảnh báo)
            model_state = torch.load(path, weights_only=True)
            self.load_state_dict(model_state)
            self.eval()  # Đặt chế độ inference
            # print(f"Model weights loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}: {e}")
            raise f"KHÔNG LOAD ĐƯỢC CRITIC"


def mean_mask(values, mask):
    mask_expanded = mask.unsqueeze(-1)  # Kích thước [batch_size, trajectory_length, 1]
    # Đếm số phần tử hợp lệ
    valid_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1)  # [batch_size, 1, 1]
    values_sum = (values * mask_expanded).sum(dim=1, keepdim=True)  # [batch_size, 1, 1]
    # Tính Mean loss của trajectory
    values_mean = values_sum / (valid_count + 1e-8)  # [batch_size, 1, 1]
    return values_mean.mean()


class PPOClip_Training:
    # Biến tĩnh tính toán số lần khởi tạo training (= số lần training)
    iter_counter = 0

    # Lưu kết quả cac lần train
    history_results = {
        "entropy": [],
        "actor_loss": [],
        "critic_loss": [],
        "mean_rewards": []
    }

    def __init__(self,
                 iters_passed,
                 actor_model,
                 critic_model,
                 states,
                 actions,
                 log_probs,
                 rewards,
                 returns,
                 advantages,
                 is_save,
                 epsilon=0.2,
                 entropy_weight=0.01,
                 num_epochs=4,
                 actor_learning_rate=0.0001,
                 critic_learning_rate=0.0001,
                 clip_value=1.0,
                 path_dir="models/param/"
                 ):
        self.iterations = iters_passed
        self.path_dir = path_dir
        self.actor = actor_model
        self.critic = critic_model
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.rewards = rewards
        self.returns = returns
        self.advantages = advantages
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.epochs = num_epochs
        self.is_save = is_save
        self.best_reward = None

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.clip_value = clip_value

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
    # Biến lớp: dùng chung cho tất cả các đối tượng
    max_epoch = 0  # số epoch tối đa
    crt_epoch = 0  # số epoch hiện tại
    train_data = []  # Danh sách lưu trữ 10 epoch gần nhất

    def train(self):

        # ====== THAY ĐỔI LEARNING RATE VÀ ENTROPY WEIGHT ==========
        # Kiểm tra xem mô hình có đang bị vướng vào điểm cực trị cục bộ
        # hoặc là không khám phá thêm được gì mới (entropy ít biến động)

        # self.entropy_weight *= 0.99 ** self.iterations  # Annealing entropy

        # ** ------------- Huấn luyện Actor -----------------**
        # Forward qua model để tính các giá trị mới
        mu, sigma, mask = self.actor(self.states)  # Tương thích với cấu trúc batch
        # print("NGOÀI _ Sigma:", sigma.min(), sigma.max(), sigma.shape)
        # Tính log-probabilities mới
        new_log_probs = compute_log_pi(self.actions, mu, sigma)  # Thêm chiều để khớp shape

        # Tính loss
        actor_loss = ppo_loss_actor(advantage=self.advantages,
                                    new_log_probs=new_log_probs,
                                    old_log_probs=self.log_probs,
                                    sigma=sigma,
                                    entropy_weight=self.entropy_weight,
                                    epsilon=self.epsilon,
                                    mask=mask)
        # Backward Actor
        self.actor_optimizer.zero_grad()
        # Tính gradient cho Actor
        actor_loss.backward()  # Tính backward để giữ đồ thị
        # Clip gradient với giá trị cụ thể
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.clip_value)

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
        critic_loss = ppo_loss_critic(predict_critic, self.returns, mask=mask)
        self.critic_optimizer.zero_grad()
        # Tính gradient cho Critic
        critic_loss.backward()  # Tính backward để giữ đồ thị
        # Clip gradient với giá trị cụ thể
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.clip_value)
        # In giá trị gradient của các tham số trong Critic
        # print("\n=== Gradient của Critic ===")
        # for name, param in self.critic.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm(2).item()  # Tính L2 norm của gradient
        #         print(f"{name} - Gradient norm: {grad_norm}")
        #     else:
        #         print(f"{name} không có gradient!")

        self.critic_optimizer.step()  # Cập nhật tham số của Critic

        # Lưu mô hình sau khi xong 1 iterator (4 epoch)
        # Tính entropy để theo dõi mức độ ngẫu nhiên của policy
        # iterations_history.append(self.it.sum(dim=1, keepdim=True)
        entropy = 0.5 * torch.sum(1 + torch.log(2 * torch.pi * sigma**2), dim=2, keepdim=True)
        PPOClip_Training.history_results["entropy"].append(mean_mask(entropy, mask).item())  # Lưu lại Entropy
        PPOClip_Training.history_results["actor_loss"].append(actor_loss.item())  # Lưu lại Actor loss
        PPOClip_Training.history_results["critic_loss"].append(critic_loss.item())  # Lưu lại Critic loss
        # Tính reward trung bình
        mean_rewards = mean_mask(self.rewards, mask).item()
        PPOClip_Training.history_results["mean_rewards"].append(mean_rewards)  # Lưu lại reward

        # =========== LƯU THÔNG TIN MÔ HÌNH SAU KHI HUẤN LUYỆN ===================

        # # So sánh phần thưởng tốt nhất và lưu mô hình nếu có cải thiện
        # if self.iterations == 0:
        #     self.best_reward = mean_reward
        # else:
        #     if self.best_reward is not None:
        #         if mean_reward > self.best_reward:
        #             self.best_reward = mean_reward
        #             best_actor_model = self.actor.state_dict()  # Lưu tham số của actor model
        #             best_critic_model = self.critic.state_dict()  # Lưu tham số của critic model
        #
        #             # Lưu mô hình tốt nhất
        #             torch.save({
        #                 'actor_model': best_actor_model,
        #                 'critic_model': best_critic_model,
        #                 'reward': self.best_reward,
        #             }, f"{self.path_dir}best_model.pth")

        # lưu mô hình Actor và Critic, sau mỗi 20 lần interation
        PPOClip_Training.iter_counter += 1
        if self.is_save:
            # if PPOClip_Training.iter_counter % 5 == 0:
            self.actor.save_model(f"{self.path_dir}actor_epoch_latest.pth")
            self.actor.save_model(f"{self.path_dir}viewer_actor_epoch_latest.pth")
            self.critic.save_model(f"{self.path_dir}critic_epoch_latest.pth")

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
                "iterations_history": int(PPOClip_Training.iter_counter),
                "rewards_history": sum(self.history_results["mean_rewards"]) / (len(self.history_results["mean_rewards"]) + 1e-6),
                "entropy_history": sum(self.history_results["entropy"]) / (len(self.history_results["entropy"]) + 1e-6),
                "actor_loss_history": sum(self.history_results["actor_loss"]) / (len(self.history_results["actor_loss"]) + 1e-6),
                "critic_loss_history": sum(self.history_results["critic_loss"]) / (len(self.history_results["critic_loss"]) + 1e-6)
            })
            for key in PPOClip_Training.history_results:
                PPOClip_Training.history_results[key] = []

            # Lưu lại toàn bộ log
            torch.save(training_log, log_file)

        return actor_loss.item(), critic_loss.item(), mean_rewards


def find_latest_model(prefix, directory="."):
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".pth")]

    if not files:
        return None

    # Sắp xếp file theo thời gian sửa đổi gần nhất
    files.sort(key=lambda f: os.path.getmtime(os.path.join(directory, f)), reverse=True)

    # Trả về file mới nhất
    return os.path.join(directory, files[0])
