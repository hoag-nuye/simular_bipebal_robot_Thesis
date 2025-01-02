# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================
import os
import pickle
import time

import torch
import numpy as np

from interface.progress_console import data_processing_console


# ========================== SUPERCLASS =================


class Buffer:
    def __init__(self):
        self.buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "log_probs": [],
            "values": [],
            "trajectory_ids": [],
        }

    def add_sample(self, state, action, reward, log_prob, value, trajectory_id):
        """Thêm một mẫu vào buffer nhỏ (RAM)."""
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["trajectory_ids"].append(trajectory_id)

    def get_samples(self):
        return self.buffer


# ========= REPLAY CÓ KÍCH THƯỚC NHỎ ĐỂ CHẠY ĐA LUỒNG =============
class ReplayCache(Buffer):
    def __init__(self):
        # Gọi hàm __init__ của class Buffer để khởi tạo buffer
        super().__init__()

    # Set id cho trajectory
    def sget_range_trajectory(self, begin_id=0):
        self.buffer["trajectory_ids"] = [_id+begin_id for _id in self.buffer["trajectory_ids"]]
        last_id = max(self.buffer["trajectory_ids"])+1
        return last_id

# ========================== RELAY BUFFER LỚN =======================


class ReplayBuffer(Buffer):
    def __init__(self,
                 trajectory_size=32 * 300,
                 max_size=50000,
                 gamma=0.99, lam=0.97, alpha=0.6,
                 file_path="replay_buffer"
                           ".pkl"):
        """
        trajectory_size: Số lượng mẫu cần thu thập mỗi lần huấn luyện.
        max_size: Kích thước tối đa của buffer khi lưu toàn bộ vào ổ cứng.
        """

        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        # Theo dõi TD error qua các lần train
        self.mean_td_error = None  # TD err hiện tại
        self.history_length = 10
        self.td_error_history = []
        self.file_path = file_path  # Đường dẫn lưu buffer vào ổ cứng

        # Gọi hàm __init__ của class Buffer để khởi tạo buffer
        super().__init__()
        # Khởi tạo thêm các giá trị cần lưu khác
        self.buffer["returns"] = []
        self.buffer["advantages"] = []
        self.buffer["td_errors"] = []

        # Kiểm tra nếu file tồn tại và xóa nó
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} đã được xóa.")
        else:
            print(f"File {file_path} không tồn tại.")

    def reset(self):
        """Reset buffer trong RAM."""
        for key in self.buffer:
            self.buffer[key] = []

    def buffer_add_sample(self, state, action, reward, log_prob, value,
                          returns, advantages, td_errors,
                          trajectory_id):
        """Thêm một mẫu vào buffer nhỏ (RAM).
        :param returns:
        :param advantages:
        :param td_errors:
        """
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["returns"].append(returns)
        self.buffer["advantages"].append(advantages)
        self.buffer["td_errors"].append(td_errors)
        self.buffer["trajectory_ids"].append(trajectory_id)

    def append_from_buffer(self, buffer):
        """
        Thêm toàn bộ dữ liệu từ buffer đầu vào vào buffer hiện tại và tính toán lại returns, advantages, TD errors.

        Args:
            buffer (dict): Buffer đầu vào phải có cấu trúc tương tự với buffer hiện tại.
        """
        # Kiểm tra xem buffer đầu vào có đầy đủ các khóa không
        required_keys = ["states", "actions", "rewards", "log_probs", "values", "trajectory_ids"]
        for key in required_keys:
            if key not in buffer:
                raise ValueError(f"Buffer đầu vào thiếu khóa: {key}")

        # Tính toán returns, advantages, và TD errors cho buffer đầu vào
        returns, advantages = self.compute_returns_and_advantages(buffer)
        td_errors = self.update_td_errors(buffer)

        # Duyệt qua từng mẫu trong buffer đầu vào và thêm vào buffer hiện tại
        for i in range(len(buffer["states"])):  # Giả định mọi danh sách đều có cùng độ dài
            self.buffer_add_sample(state=buffer["states"][i],
                                   action=buffer["actions"][i],
                                   reward=buffer["rewards"][i],
                                   log_prob=buffer["log_probs"][i],
                                   value=buffer["values"][i],
                                   returns=returns[i],
                                   advantages=advantages[i],
                                   td_errors=td_errors[i],
                                   trajectory_id=buffer["trajectory_ids"][i])

    # =========================== TÍNH At và Gt ========================
    def compute_returns_and_advantages(self, buffer):
        """Tính toán returns và advantages cho buffer đầu vào theo GAE (\u03bb=1)."""
        T = len(buffer["rewards"])
        returns = [0] * T
        advantages = [0] * T

        next_value = 0  # Giá trị V(s_T) (ở ngoài trajectory)
        next_advantage = 0

        for t in reversed(range(T)):
            # Kiểm tra ngắt trajectory
            if t == T - 1 or (buffer["trajectory_ids"][t] != buffer["trajectory_ids"][t + 1]):
                next_value = 0  # Reset giá trị V(s) khi ngắt trajectory
                next_advantage = 0

            delta = buffer["rewards"][t] + self.gamma * next_value - buffer["values"][t]
            advantages[t] = delta + self.gamma * self.lam * next_advantage
            returns[t] = advantages[t] + buffer["values"][t]

            next_value = buffer["values"][t]
            next_advantage = advantages[t]
        return returns, advantages

    #  =============== Tính toán TD errors =====================
    def update_td_errors(self, buffer):
        """Tính toán TD-error cho buffer đầu vào."""
        td_errors = []
        for t in range(len(buffer["rewards"])):
            reward = buffer["rewards"][t]
            value = buffer["values"][t]
            next_value = buffer["values"][t + 1] if t + 1 < len(buffer["rewards"]) else 0
            td_error = reward + self.gamma * next_value - value
            td_errors.append(td_error)

        self.mean_td_error = np.mean(td_errors)
        self.td_error_history.append(self.mean_td_error)
        # # --------- KIỂM TRA VIỆC TÍNH TD ERR -----------
        # td_err = []
        # r = []
        # v = []
        # for idx, t in enumerate(buffer["trajectory_ids"]):
        #     if t == 0:
        #         r.append(np.array(buffer["rewards"])[idx])
        #         v.append(np.array(buffer["values"])[idx])
        # print(f"td_er_calculator: {td_errors}")
        # print(f"reward: {r}")
        # print(f"values: {v}")
        return td_errors

    # == kiểm tra over-under fitting--
    def detect_overfitting_or_underfitting(self):
        """
        Phát hiện overfitting hoặc underfitting dựa trên TD-error trung bình và lịch sử.
        - Overfitting: TD-error giảm mạnh và ổn định (giá trị rất thấp so với lịch sử).
        - Underfitting: TD-error tăng đều hoặc không giảm trong lịch sử.
        """
        # So sánh TD-error hiện tại với lịch sử
        if len(self.td_error_history) > 1:
            mean_error = np.mean(self.td_error_history)
            std_error = np.std(self.td_error_history)
            normalized_error = (self.mean_td_error - mean_error) / (std_error + 1e-6)

            # print(normalized_error)

            if normalized_error < -2:
                return "overfitting"
            elif normalized_error > 2:
                return "underfitting"

        return "normal"

    # ====================== TẠO RA CÁC MINIBATCH ================

    # ====================== LẤY DỮ LIỆU CHO TRAINING ============
    # SỬ DỤNG CHO NHỮNG THUẬT TOÁN ON-POLICY (HUẤN LUYỆN TỪ STATE HIỆN TẠI)
    def sample_batch(self, batch_size=32,):
        mini_batch_dict = []

        # Tạo các nhóm tương ứng cho trajectory_ids với chỉ mục
        trajectory_ids = self.buffer["trajectory_ids"]

        # Lấy các giá trị duy nhất từ trajectory_ids và random hóa
        unique_trajectories = np.unique(trajectory_ids)
        # print(unique_trajectories)
        if batch_size > len(unique_trajectories):
            raise ValueError(f"Batch size ({batch_size}) lớn hơn số trajectory khả dụng ({len(unique_trajectories)}).")

        # Xáo trộn các giá trị duy nhất
        # np.random.shuffle(unique_trajectories)

        # Tạo danh sách chứa các nhóm với mỗi nhóm có len_batch giá trị duy nhất của buffer
        grouped_trajectories = [unique_trajectories[i:i + batch_size].tolist() for i in
                                range(0, len(unique_trajectories), batch_size)]
        # print(grouped_trajectories)
        # Lấy các chỉ mục tương ứng được chia theo grouped_trajectories
        batches_indices = []
        grouped_trajectories_size = len(grouped_trajectories)
        start_time = time.time()
        for idx, group in enumerate(grouped_trajectories):
            data_processing_console(total_steps=grouped_trajectories_size,
                                    current_steps=idx + 1,
                                    begin_time=start_time)
            group_indices = np.where(np.isin(trajectory_ids, group))[0]  # Chỉ mục tương ứng với group
            batches_indices.append(group_indices)

        # Lấy các giá trị trong buffer theo các chỉ mục đã chọn
        batches_indices_size = len(batches_indices)
        start_time = time.time()
        # Chuyển self.buffer sang NumPy array một lần duy nhất
        numpy_buffer = {key: np.array(value) for key, value in self.buffer.items()}

        for idx, batch_indices in enumerate(batches_indices):
            data_processing_console(total_steps=batches_indices_size,
                                    current_steps=idx + 1,
                                    begin_time=start_time)

            # Truy xuất dữ liệu nhanh hơn từ numpy_buffer
            batch_dict = {key: numpy_buffer[key][batch_indices] for key in numpy_buffer}
            mini_batch_dict.append(batch_dict)

        # for idx, batch_indices in enumerate(batches_indices):
        #     data_processing_console(total_steps=batches_indices_size,
        #                             current_steps=idx + 1,
        #                             begin_time=start_time)
        #     batch_dict = {key: np.array([]) for key in buffer.keys()}
        #     for key in buffer.keys():
        #         batch_dict[key] = np.array(buffer[key])[batch_indices]
        #     mini_batch_dict.append(batch_dict)

        # ===== TẠO ĐẦU VÀO CHO VIỆC HUẤN LUYỆN THEO CÁC BATCH ĐÃ CHIA ==============
        """
        ĐẦU RA LÀ CÁC BATCH CÓ CHỨA DỮ LIỆU CỦA CÁC
        TENSOR CÓ SHAPE LÀ : [batch_size, num_samples, feature_size]
        """
        mini_batch = []
        len_mini_batch_dict = len(mini_batch_dict)
        start_time = time.time()
        for idx, _batch_dict in enumerate(mini_batch_dict):
            data_processing_console(total_steps=len_mini_batch_dict,
                                    current_steps=idx+1,
                                    begin_time=start_time)
            # Khởi tạo dictionary chứa các batch đã được chia
            batch = {}
            # Khởi tạo giá trị lưu độ dài thực tế của từng traj trong batch
            lengths_traj = None
            # Duyệt qua tất cả các keys và chuyển đổi dữ liệu thành tensor
            trajectory_ids = _batch_dict["trajectory_ids"]  # Dữ liệu trajectory_ids
            # print(len(np.unique(trajectory_ids)))
            # print(np.unique(trajectory_ids))
            for key in _batch_dict.keys():
                if key != "trajectory_ids":
                    unique_trajectories = np.unique(trajectory_ids)
                    # Lấy liệu cho key hiện tại
                    data = _batch_dict[key]
                    # if key == "rewards":
                    #     for idx, d in enumerate(data):
                    #         if _batch_dict["trajectory_ids"][idx] == _batch_dict["trajectory_ids"][1]:
                    #             print(data[idx])
                    # Tạo một mảng chứa mask cho mỗi trajectory_id
                    mask_list = [trajectory_ids == trajectory_id for trajectory_id in unique_trajectories]

                    # Dùng broadcasting và slicing để nhóm dữ liệu theo trajectory_id
                    grouped_data = [data[mask] for mask in mask_list]

                    # Tìm số mẫu lớn nhất trong mỗi nhóm để padding
                    max_samples = max(len(group) for group in grouped_data)

                    # lengths_traj: Danh sách độ dài thực của từng trajectory trong batch
                    lengths_traj = torch.tensor([len(group) for group in grouped_data])

                    # ĐƯA VỀ ARR 1 SANG VỀ 2 CHIỀU ĐỂ PADDING
                    for i, group in enumerate(grouped_data):
                        if len(group.shape) != 2:
                            grouped_data[i] = group.reshape(-1, 1)  # Gán lại group vào grouped_data

                    # Padding dữ liệu sao cho tất cả các nhóm đều có số lượng mẫu như nhau
                    # (0, max_samples - len(group)), (0, 0) => mảng được padding phải là mảng 2 chiều
                    padded_data = np.array(
                        [np.pad(group,
                                ((0, max_samples - len(group)), (0, 0)),
                                mode='constant') for group in grouped_data])

                    # Chuyển thành tensor
                    batch[key] = torch.tensor(padded_data)
                    batch["lengths_traj"] = lengths_traj
            mini_batch.append(batch)
        return mini_batch

    # SỬ DỤNG CHO NHỮNG THUẬT TOÁN OFF-POLICY (HUẤN LUYỆN TỪ CÁC DỮ LIỆU KHÁC)
    # def sample_batch_use_buffer(self, batch_size=32):
    #     """
    #     Sampling batch dựa trên trạng thái overfitting hoặc underfitting, từ buffer lớn được tải từ ổ cứng.
    #     """
    #     batch_result = {}  # lưu kết quả
    #     lengths_traj = None  # lưu kết quả vị trí các sample và sampling
    #     selected_trajectory_ids = []
    #     # Tải dữ liệu từ ổ cứng (buffer lớn)
    #     buffer_large = self.load_from_pkl()
    #
    #     # 1. Lấy danh sách các giá trị duy nhất trong "trajectory_ids"
    #     unique_trajectory_ids = np.unique(buffer_large["trajectory_ids"])
    #     num_trajectories = len(unique_trajectory_ids)
    #     # Đảm bảo batch size không vượt quá số trajectory có sẵn
    #     if batch_size > num_trajectories:
    #         raise ValueError(f"Batch size ({batch_size}) lớn hơn số trajectory khả dụng ({num_trajectories}).")
    #
    #     # 2. Chuyển dữ liệu về array
    #     batch_dict = {
    #         "states": np.array(buffer_large["states"]),  # 2D
    #         "actions": np.array(buffer_large["actions"]),  # 2D
    #         "log_probs": np.array(buffer_large["log_probs"]).reshape(-1, 1),  # 1D -> 2D
    #         "rewards": np.array(buffer_large["rewards"]).reshape(-1, 1),  # 1D -> 2D
    #         "returns": np.array(buffer_large["returns"]).reshape(-1, 1),  # 1D -> 2D
    #         "advantages": np.array(buffer_large["advantages"]).reshape(-1, 1),  # 1D -> 2D
    #         "td_errors": np.array(buffer_large["td_errors"]).reshape(-1, 1),  # 1D -> 2D
    #         # 1D -> 2D
    #         "trajectory_ids": np.array(buffer_large["trajectory_ids"])}
    #
    #     # print(batch_dict["trajectory_ids"].shape, batch_dict["states"].shape, np.array(buffer_large["states"]).shape)
    #
    #     # 3. Khởi tạo dictionary kết quả
    #     batch_large = {}
    #     # Duyệt qua tất cả các keys và chuyển đổi dữ liệu thành tensor
    #     trajectory_ids = batch_dict["trajectory_ids"]  # Dữ liệu trajectory_ids
    #     for key in batch_dict.keys():
    #         if key != "trajectory_ids":
    #
    #             data = np.array(batch_dict[key])  # Chuyển dữ liệu sang mảng NumPy nếu cần  # Dữ liệu cho key hiện tại
    #             # print(key, batch_dict[key].shape)
    #             # Tạo một mảng chứa mask cho mỗi trajectory_id
    #             mask_list = np.array([trajectory_ids == trajectory_id for trajectory_id in unique_trajectory_ids])
    #             # print(data.shape, mask_list.shape)
    #
    #             grouped_data = [data[mask] for mask in mask_list]
    #
    #             # Tìm số mẫu lớn nhất trong mỗi nhóm để padding
    #             max_samples = max(len(group) for group in grouped_data)
    #
    #             # # Kiểm tra kích thước của từng group trước khi padding
    #             # print(f"Max samples: {max_samples}")
    #             # print(f"Grouped data shapes before padding: {[group.shape for group in grouped_data]}")
    #
    #             # Padding dữ liệu sao cho tất cả các nhóm đều có số lượng mẫu như nhau
    #             padded_data = np.array(
    #                 [np.pad(group, ((0, max_samples - len(group)), (0, 0)), mode='constant', constant_values=-1) for
    #                  group in grouped_data])
    #
    #             # Chuyển thành tensor
    #             batch_large[key] = torch.tensor(padded_data)
    #
    #     # print(batch_large["states"].shape)
    #     # print("TENSOR:", result["states"][-1, 0, :])
    #     # print("DEQUE:", [buffer_large["states"][i] for i in range(len(buffer_large["trajectory_ids"]))
    #     #                  if buffer_large["trajectory_ids"][i] == unique_trajectory_ids[-1]])
    #     # ============================================================
    #     # =================== LỌC DỮ LIỆU ============================
    #     # =============================================================
    #
    #     # Phát hiện trạng thái huấn luyện
    #     status = self.detect_overfitting_or_underfitting()
    #
    #     # ========== LẤY DỮ LIỆU KHI GẶP PHẢI OVER-UNDER FIT================
    #     if status in ["overfitting", "underfitting"]:
    #         # print(status)
    #         # 90% trajectory lấy từ dữ liệu mới nhất
    #         percent = 0.9
    #         batch_main_size = int(batch_size * percent)
    #         bath_rest_size = int(batch_size - batch_main_size)  # Số lượng cần lấy
    #
    #         main_trajectory_ids = unique_trajectory_ids[-batch_main_size:]
    #         rest_trajectory_ids = unique_trajectory_ids[:-batch_main_size]
    #
    #         # print(main_trajectory_ids, rest_trajectory_ids)
    #
    #         if batch_main_size != batch_size:
    #             batch_main = {key: tensor[torch.tensor(main_trajectory_ids)] for key, tensor in batch_large.items()}
    #             rest_trajectory = {key: tensor[torch.tensor(rest_trajectory_ids)] for key, tensor in
    #                                batch_large.items()}  # dữ liệu được phép lấy
    #             # Bước 1: Truy xuất rewards và tính priorities
    #             td_errors = rest_trajectory["td_errors"]  # ex: Shape: [32, 300, 1]
    #             mean_td_errors = td_errors.mean(dim=1, keepdim=False)  # Shape: [32, 1]
    #             # TD Error (𝛿) có thể dương hoặc âm
    #             # nhưng chỉ cần quan tâm đến độ lớn của sai lệch (absolute difference), không cần quan tâm đến dấu.
    #             priorities = torch.abs(mean_td_errors).squeeze(
    #                 -1) ** self.alpha  # Shape: [32] # priorities mũ lên để thể hiện sự quan trọng (ưu tiên)
    #             # Bước 2: Normalize probabilities
    #             probabilities = priorities / torch.sum(priorities)
    #             # Đảm bảo tổng xác suất = 1 bằng cách chuẩn hóa lại nếu cần
    #             if not np.isclose(probabilities.sum().item(), 1.0, atol=1e-6):
    #                 probabilities = probabilities / probabilities.sum()
    #             # Bước 3: Chọn num_choice chỉ mục theo xác suất ưu tiên
    #             selected_indices = np.random.choice(
    #                 rest_trajectory_ids, size=bath_rest_size, replace=False, p=probabilities)
    #             # Bước 4: Trích xuất tensor tương ứng từ result
    #             batch_rest = {key: value[selected_indices] for key, value in rest_trajectory.items()}
    #             # Bước 5: Gộp batch_main và batch_rest
    #             selected_trajectory_ids = np.concatenate((selected_indices, main_trajectory_ids))
    #             batch_result = {key: torch.cat([batch_main[key], batch_rest[key]], dim=0)
    #                             for key in batch_large.keys()}
    #
    #         else:
    #             # Tạo dictionary mới chứa các tensor đã lọc
    #             selected_trajectory_ids = main_trajectory_ids
    #             batch_result = {key: tensor[torch.tensor(main_trajectory_ids)] for key, tensor in batch_large.items()}
    #     # ========== LẤY DỮ LIỆU NHƯ BÌNH THƯỜNG================
    #     else:
    #         # print(status)
    #         # Nếu overfitting hoặc underfitting:
    #         # 90% trajectory lấy từ dữ liệu mới nhất
    #         percent = 0.5
    #         batch_main_size = int(batch_size * percent)
    #         bath_rest_size = int(batch_size - batch_main_size)  # Số lượng cần lấy
    #
    #         main_trajectory_ids = unique_trajectory_ids[-batch_main_size:]
    #         rest_trajectory_ids = unique_trajectory_ids[:-bath_rest_size]
    #
    #         if batch_main_size != batch_size:
    #             batch_main = {key: tensor[torch.tensor(main_trajectory_ids)] for key, tensor in batch_large.items()}
    #             rest_trajectory = {key: tensor[torch.tensor(rest_trajectory_ids)] for key, tensor in
    #                                batch_large.items()}  # dữ liệu được phép lấy
    #             # Bước 1: Truy xuất reward và tính priorities
    #             rewards = rest_trajectory["rewards"]  # Shape: [32, 300, 1]
    #             mean_rewards = rewards.mean(dim=1, keepdim=False)  # Shape: [32, 1]
    #             # đảm bảo phần thưởng trung bình dương nhưng vẫn xếp theo thứ tự
    #             min_reward = mean_rewards.min().item()  # Giá trị phần thưởng nhỏ nhất
    #             epsilon = 1e-6  # Giá trị nhỏ để tránh 0
    #             adjusted_rewards = mean_rewards - min_reward + 1e-6  # Đảm bảo không có giá trị âm
    #             priorities = adjusted_rewards.squeeze(-1) ** 2  # Shape: [32]
    #             # Bước 2: Normalize probabilities
    #             probabilities = priorities / torch.sum(priorities)
    #             # Đảm bảo tổng xác suất = 1 bằng cách chuẩn hóa lại nếu cần
    #             if not np.isclose(probabilities.sum().item(), 1.0, atol=1e-6):
    #                 probabilities = probabilities / probabilities.sum()
    #             # Bước 3: Chọn num_choice chỉ mục theo xác suất ưu tiên
    #             selected_indices = np.random.choice(
    #                 rest_trajectory_ids, size=bath_rest_size, replace=False, p=probabilities)
    #             # Bước 4: Trích xuất tensor tương ứng từ result
    #             batch_rest = {key: value[selected_indices] for key, value in rest_trajectory.items()}
    #             # Bước 5: Gộp batch_main và batch_rest
    #             selected_trajectory_ids = np.concatenate((selected_indices, main_trajectory_ids))
    #             batch_result = {key: torch.cat([batch_main[key], batch_rest[key]], dim=0)
    #                             for key in batch_large.keys()}
    #         else:
    #             # Tạo dictionary mới chứa các tensor đã lọc
    #             selected_trajectory_ids = main_trajectory_ids
    #             batch_result = {key: tensor[torch.tensor(main_trajectory_ids)] for key, tensor in batch_large.items()}
    #
    #     # ============================================================
    #     # =================== XỬ LÝ LẠI PADDING ======================
    #     # ============================================================
    #     # --------------- TÌM LENGHT THẬT CỦA TRAJECTORY ------
    #     # Tạo một mask để lọc lại dữ liệu dựa trên các trajectory_id trong batch_result
    #     mask_list_result = [batch_dict["trajectory_ids"] == traj_id for traj_id in selected_trajectory_ids]
    #
    #     # Sử dụng mask để nhóm lại dữ liệu trong batch_dict (hoặc batch_large)
    #     grouped_data_result = [batch_dict["states"][mask] for mask in
    #                            mask_list_result]  # Có thể thay 'states' bằng key khác
    #
    #     # Tính lại độ dài thực sự của mỗi trajectory trong batch_result
    #     lengths_traj = torch.tensor([len(group) for group in grouped_data_result])
    #     # ---------- TÌM SIZE PADDING THẬT CỦA SAMPLE --------------
    #     max_lengths_traj = torch.max(lengths_traj)
    #     for key in batch_result.keys():
    #         batch_result[key] = batch_result[key][:, :max_lengths_traj, :]
    #     print(batch_result["states"].shape)
    #     # print(lengths_traj)
    #     return batch_result, lengths_traj
    #
    # # =================== LƯU TRỮ DỮ LIỆU =====================
    # def save_to_pkl(self):
    #     """Lưu buffer vào file .pkl, xóa mẫu cũ nhất nếu vượt quá max_size."""
    #     temp_file_path = self.file_path + ".tmp"
    #     current_buffer_size = len(self.buffer["states"])  # Số mẫu trong RAM
    #
    #     if os.path.exists(self.file_path):
    #         print(f"File {self.file_path} tồn tại.")
    #
    #     # Ghi dữ liệu mới vào file tạm
    #     with open(temp_file_path, "wb") as temp_file:
    #         samples_written = 0
    #         try:
    #             # Đọc file gốc
    #             with open(self.file_path, "rb") as original_file:
    #                 while True:
    #                     try:
    #                         # Load một phần dữ liệu từ file gốc
    #                         data = pickle.load(original_file)
    #                         # Xóa phần tử cũ nếu vượt quá max_size
    #                         for key in data:
    #                             while len(data[key]) > 0 and samples_written + len(
    #                                     data[key]) > self.max_size - current_buffer_size:
    #                                 data[key].popleft()
    #                         # Ghi phần dữ liệu còn lại vào file tạm
    #                         pickle.dump(data, temp_file)
    #                         samples_written += len(data["states"])
    #                     except EOFError:
    #                         break
    #         except FileNotFoundError:
    #             print("File gốc không tồn tại, tạo file mới.")
    #
    #         # Ghi buffer nhỏ (trong RAM) vào file tạm
    #         pickle.dump(self.buffer, temp_file)
    #
    #     # Thay thế file gốc bằng file tạm
    #     os.replace(temp_file_path, self.file_path)
    #
    # def load_from_pkl(self):
    #     """Tải toàn bộ dữ liệu từ file .pkl."""
    #     buffer_large = {
    #         "states": [],
    #         "actions": [],
    #         "rewards": [],
    #         "log_probs": [],
    #         "returns": [],
    #         "advantages": [],
    #         "trajectory_ids": [],
    #         "td_errors": [],
    #     }
    #     try:
    #         with open(self.file_path, "rb") as f:
    #             while True:
    #                 try:
    #                     data = pickle.load(f)
    #                     for key in buffer_large:
    #                         buffer_large[key].extend(data[key])
    #                 except EOFError:
    #                     break
    #     except FileNotFoundError:
    #         print("File not found, starting with an empty buffer.")
    #     return buffer_large
