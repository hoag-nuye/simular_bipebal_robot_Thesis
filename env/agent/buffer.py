import os
import pickle
import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self,
                 trajectory_size=32*300,
                 max_size=50000,
                 gamma=0.99, lam=0.95, alpha=0.6,
                 file_path="replay_buffer"
                                                                                                      ".pkl"):
        """
        trajectory_size: Số lượng mẫu cần thu thập mỗi lần huấn luyện.
        max_size: Kích thước tối đa của buffer khi lưu toàn bộ vào ổ cứng.
        """
        self.trajectory_size = trajectory_size  # Kích thước của buffer trong mỗi lần thu thập
        self.max_size = max_size  # Kích thước tối đa khi lưu trữ
        self.buffer = {
            "states": deque(maxlen=trajectory_size),
            "actions": deque(maxlen=trajectory_size),
            "rewards": deque(maxlen=trajectory_size),
            "next_states": deque(maxlen=trajectory_size),
            "log_probs": deque(maxlen=trajectory_size),
            "values": deque(maxlen=trajectory_size),
            "timesteps": deque(maxlen=trajectory_size),
            "trajectory_ids": deque(maxlen=trajectory_size),
            "td_errors": deque(maxlen=trajectory_size),  # Chỉ giữ TD-error cho trajectory_size
        }
        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.mean_td_error = 0
        self.file_path = file_path  # Đường dẫn lưu buffer vào ổ cứng

        # Kiểm tra nếu file tồn tại và xóa nó
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} đã được xóa.")
        else:
            print(f"File {file_path} không tồn tại.")

    def add_sample(self, state, action, reward, next_state, log_prob, value, timestep, trajectory_id):
        """Thêm một mẫu vào buffer nhỏ (RAM)."""
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["next_states"].append(next_state)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["timesteps"].append(timestep)
        self.buffer["trajectory_ids"].append(trajectory_id)
        self.buffer["td_errors"].append(0)


    def update_td_errors(self):
        """Cập nhật TD-error trong buffer nhỏ."""
        td_errors = []
        for t in range(len(self.buffer["rewards"])):
            reward = self.buffer["rewards"][t]
            value = self.buffer["values"][t]
            next_value = self.buffer["values"][t + 1] if t + 1 < len(self.buffer["rewards"]) else 0
            td_error = abs(reward + self.gamma * next_value - value)
            td_errors.append(td_error)

        self.buffer["td_errors"] = deque(td_errors, maxlen=self.trajectory_size)
        # print(np.array(self.buffer["td_errors"]).shape,
        #       np.array(self.buffer["rewards"]).shape,
        #       np.array(self.buffer["values"]).shape)
        self.mean_td_error = np.mean(td_errors)

    def compute_returns_and_advantages(self):
        """Tính toán returns và advantages trong buffer nhỏ."""
        returns = []
        advantages = []
        G = 0
        advantage = 0

        for t in reversed(range(len(self.buffer["rewards"]))):
            if (t == len(self.buffer["rewards"]) - 1 or
                    self.buffer["trajectory_ids"][t] != self.buffer["trajectory_ids"][t + 1]):
                G = self.buffer["values"][t]
                advantage = 0

            next_value = 0 if t == len(self.buffer["rewards"]) - 1 else self.buffer["values"][t + 1]
            delta = self.buffer["rewards"][t] + self.gamma * next_value - self.buffer["values"][t]
            advantage = delta + self.gamma * self.lam * advantage
            advantages.insert(0, advantage)

            G = advantage + self.buffer["values"][t]
            returns.insert(0, G)

        self.buffer["returns"] = deque(returns, maxlen=self.trajectory_size)
        self.buffer["advantages"] = deque(advantages, maxlen=self.trajectory_size)

    def save_to_pkl(self):
        """Lưu buffer vào file .pkl."""
        with open(self.file_path, "ab") as f:  # Mở file ở chế độ append binary
            pickle.dump(self.buffer, f)

    def load_from_pkl(self):
        """Tải toàn bộ dữ liệu từ file .pkl."""
        buffer_large = {
            "states": deque(maxlen=self.max_size),
            "actions": deque(maxlen=self.max_size),
            "rewards": deque(maxlen=self.max_size),
            "next_states": deque(maxlen=self.max_size),
            "log_probs": deque(maxlen=self.max_size),
            "values": deque(maxlen=self.max_size),
            "timesteps": deque(maxlen=self.max_size),
            "trajectory_ids": deque(maxlen=self.max_size),
            "td_errors": deque(maxlen=self.max_size),
        }
        try:
            with open(self.file_path, "rb") as f:
                while True:
                    try:
                        data = pickle.load(f)
                        for key in buffer_large:
                            buffer_large[key].extend(data[key])
                    except EOFError:
                        break
        except FileNotFoundError:
            print("File not found, starting with an empty buffer.")
        return buffer_large

    def detect_overfitting_or_underfitting(self, threshold=0.05):
        """
        Phát hiện overfitting hoặc underfitting dựa trên TD-error trung bình.
        - Overfitting: TD-error giảm đáng kể (giá trị rất thấp).
        - Underfitting: TD-error tăng đều hoặc không ổn định.
        """
        if self.mean_td_error < threshold:
            return "overfitting"
        elif self.mean_td_error > 2 * threshold:
            return "underfitting"
        return "normal"

    def sample_batch(self, batch_size=32):
        """
        Sampling batch dựa trên trạng thái overfitting hoặc underfitting, từ buffer lớn được tải từ ổ cứng.
        """
        # Phát hiện trạng thái huấn luyện
        status = self.detect_overfitting_or_underfitting()

        # Tải dữ liệu từ ổ cứng (buffer lớn)
        buffer_large = self.load_from_pkl()

        if status in ["overfitting", "underfitting"]:
            # Nếu overfitting hoặc underfitting:
            # 90% mẫu lấy từ dữ liệu mới nhất
            new_indices = np.arange(len(buffer_large["trajectory_ids"]))[-batch_size * 9 // 10:]
            # 10% mẫu lấy ngẫu nhiên theo trọng số ưu tiên từ toàn bộ buffer
            td_errors = np.array(buffer_large["td_errors"])
            priorities = td_errors ** self.alpha
            probabilities = priorities / priorities.sum()

            old_indices = np.random.choice(
                len(buffer_large["trajectory_ids"]),
                size=batch_size * 1 // 10,
                p=probabilities,
                replace=False
            )
            selected_indices = np.concatenate([new_indices, old_indices])
        else:
            # Nếu trạng thái bình thường:
            # 100% mẫu lấy từ dữ liệu mới nhất
            selected_indices = np.arange(len(buffer_large["trajectory_ids"]))[-batch_size:]

        # Shuffle để đảm bảo tính ngẫu nhiên
        np.random.shuffle(selected_indices)

        # Trả về batch dữ liệu
        return {
            "states": np.array([buffer_large["states"][i] for i in selected_indices]),
            "actions": np.array([buffer_large["actions"][i] for i in selected_indices]),
            "rewards": np.array([buffer_large["rewards"][i] for i in selected_indices]),
            "next_states": np.array([buffer_large["next_states"][i] for i in selected_indices]),
            "log_probs": np.array([buffer_large["log_probs"][i] for i in selected_indices]),
            "values": np.array([buffer_large["values"][i] for i in selected_indices]),
            "trajectory_ids": np.array([buffer_large["trajectory_ids"][i] for i in selected_indices]),
        }

    def reset(self):
        """Reset buffer nhỏ trong RAM."""
        for key in self.buffer:
            self.buffer[key] = deque(maxlen=self.trajectory_size)
