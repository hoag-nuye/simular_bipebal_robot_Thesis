import os
import pickle
import torch
import numpy as np
from collections import deque, defaultdict

# ========================== SUPERCLASS =================


class Buffer:
    def __init__(self, trajectory_size=32 * 300,
                 max_size=50000):
        self.trajectory_size = trajectory_size  # Kích thước của buffer trong mỗi lần thu thập
        self.max_size = max_size  # Kích thước tối đa khi lưu trữ
        self.buffer = {
            "states": deque(maxlen=trajectory_size),
            "actions": deque(maxlen=trajectory_size),
            "rewards": deque(maxlen=trajectory_size),
            "log_probs": deque(maxlen=trajectory_size),
            "sigma": deque(maxlen=trajectory_size),
            "values": deque(maxlen=trajectory_size),
            "trajectory_ids": deque(maxlen=trajectory_size),
            "td_errors": deque(maxlen=trajectory_size),  # Chỉ giữ TD-error cho trajectory_size
        }

    def add_sample(self, state, action, reward, log_prob, sigma, value, trajectory_id):
        """Thêm một mẫu vào buffer nhỏ (RAM)."""
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["sigma"].append(sigma)
        self.buffer["values"].append(value)
        self.buffer["trajectory_ids"].append(trajectory_id)

    def get_samples(self):
        return self.buffer

# ========================== RELAY BUFFER LỚN =======================


class ReplayBuffer(Buffer):
    def __init__(self,
                 trajectory_size=32 * 300,
                 max_size=50000,
                 gamma=0.99, lam=0.95, alpha=0.6,
                 file_path="replay_buffer"
                           ".pkl"):
        """
        trajectory_size: Số lượng mẫu cần thu thập mỗi lần huấn luyện.
        max_size: Kích thước tối đa của buffer khi lưu toàn bộ vào ổ cứng.
        """

        self.gamma = gamma
        self.lam = lam
        self.alpha = alpha
        self.mean_td_error = 0
        self.file_path = file_path  # Đường dẫn lưu buffer vào ổ cứng

        # Gọi hàm __init__ của class Buffer để khởi tạo buffer
        super().__init__(trajectory_size, max_size)

        # Kiểm tra nếu file tồn tại và xóa nó
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} đã được xóa.")
        else:
            print(f"File {file_path} không tồn tại.")

    def append_from_buffer(self, buffer):
        """
        Thêm toàn bộ dữ liệu từ buffer đầu vào vào buffer hiện tại.

        Args:
            buffer (dict): Buffer đầu vào phải có cấu trúc tương tự với buffer hiện tại.
        """
        # Kiểm tra xem buffer đầu vào có đầy đủ các khóa không
        required_keys = ["states", "actions", "rewards", "log_probs", "sigma", "values", "trajectory_ids"]
        for key in required_keys:
            if key not in buffer:
                raise ValueError(f"Buffer đầu vào thiếu khóa: {key}")

        # Duyệt qua từng mẫu trong buffer đầu vào và thêm vào buffer hiện tại
        for i in range(len(buffer["states"])):  # Giả định mọi danh sách đều có cùng độ dài
            self.add_sample(
                state=buffer["states"][i],
                action=buffer["actions"][i],
                reward=buffer["rewards"][i],
                log_prob=buffer["log_probs"][i],
                sigma=buffer["sigma"][i],
                value=buffer["values"][i],
                trajectory_id=buffer["trajectory_ids"][i],
            )

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
        """Lưu buffer vào file .pkl, xóa mẫu cũ nhất nếu vượt quá max_size."""
        temp_file_path = self.file_path + ".tmp"
        current_buffer_size = len(self.buffer["states"])  # Số mẫu trong RAM

        if os.path.exists(self.file_path):
            print(f"File {self.file_path} tồn tại.")

        # Ghi dữ liệu mới vào file tạm
        with open(temp_file_path, "wb") as temp_file:
            samples_written = 0
            try:
                # Đọc file gốc
                with open(self.file_path, "rb") as original_file:
                    while True:
                        try:
                            # Load một phần dữ liệu từ file gốc
                            data = pickle.load(original_file)
                            # Xóa phần tử cũ nếu vượt quá max_size
                            print(len(data["states"]), self.max_size - current_buffer_size)
                            for key in data:
                                while len(data[key]) > 0 and samples_written + len(
                                        data[key]) > self.max_size - current_buffer_size:
                                    data[key].popleft()
                            # Ghi phần dữ liệu còn lại vào file tạm
                            pickle.dump(data, temp_file)
                            samples_written += len(data["states"])
                        except EOFError:
                            break
            except FileNotFoundError:
                print("File gốc không tồn tại, tạo file mới.")

            # Ghi buffer nhỏ (trong RAM) vào file tạm
            pickle.dump(self.buffer, temp_file)

        # Thay thế file gốc bằng file tạm
        os.replace(temp_file_path, self.file_path)

        # # Kiểm tra xem có thêm được hay không
        # try:
        #     with open(self.file_path, "rb") as f:
        #         while True:
        #             try:
        #                 data = pickle.load(f)
        #                 print(len(data["states"]))
        #             except EOFError:
        #                 break
        # except FileNotFoundError:
        #     print("File not found, starting with an empty buffer.")

    def load_from_pkl(self):
        """Tải toàn bộ dữ liệu từ file .pkl."""
        buffer_large = {
            "states": deque(maxlen=self.max_size),
            "actions": deque(maxlen=self.max_size),
            "rewards": deque(maxlen=self.max_size),
            "log_probs": deque(maxlen=self.max_size),
            "sigma": deque(maxlen=self.max_size),
            "returns": deque(maxlen=self.max_size),
            "advantages": deque(maxlen=self.max_size),
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

        # Tính danh sách các trajectory id duy nhất
        unique_trajectory_ids = np.unique(buffer_large["trajectory_ids"])
        num_trajectories = len(unique_trajectory_ids)

        # Đảm bảo batch size không vượt quá số trajectory có sẵn
        if batch_size > num_trajectories:
            raise ValueError(f"Batch size ({batch_size}) lớn hơn số trajectory khả dụng ({num_trajectories}).")

        # Chọn các trajectory id để tạo batch
        if status in ["overfitting", "underfitting"]:
            # Nếu overfitting hoặc underfitting:
            # 90% trajectory lấy từ dữ liệu mới nhất
            new_trajectory_ids = unique_trajectory_ids[-batch_size * 9 // 10:]

            if not (len(new_trajectory_ids) == batch_size):
                # 10% trajectory lấy ngẫu nhiên theo trọng số ưu tiên
                td_errors = np.array(buffer_large["td_errors"])
                priorities = td_errors ** self.alpha  # priorities mũ lên để thể hiện sự quan trọng (ưu tiên)
                probabilities = priorities / priorities.sum()  # Normalize probabilities

                # Tính probabilities cho từng trajectory_id
                trajectory_probabilities = defaultdict(float)  # Dictionary để lưu probabilities theo trajectory_id

                # Lặp qua tất cả các sample và gộp probabilities theo trajectory_id
                for prob, traj_id in zip(probabilities, buffer_large["trajectory_ids"]):
                    trajectory_probabilities[traj_id] += prob

                # Chuyển các giá trị trong dict thành mảng NumPy
                trajectory_probabilities_array = np.array(list(trajectory_probabilities.values()))

                trajectory_probabilities_array = trajectory_probabilities_array[:-len(new_trajectory_ids)]

                # Nếu tổng không chính xác bằng 1, chuẩn hóa lại
                if not np.isclose(trajectory_probabilities_array.sum(), 1.0):
                    trajectory_probabilities_array /= trajectory_probabilities_array.sum()

                old_trajectory_ids = np.random.choice(
                    unique_trajectory_ids[:-len(new_trajectory_ids)],  # Trừ các trajectory mới nhất
                    size=batch_size * 1 // 10,
                    p=trajectory_probabilities_array,
                    replace=False
                )
                selected_trajectory_ids = np.concatenate([new_trajectory_ids, old_trajectory_ids])
            else:
                selected_trajectory_ids = unique_trajectory_ids[-batch_size:]
        else:
            # Nếu trạng thái bình thường:
            # 100% trajectory lấy từ dữ liệu mới nhất
            selected_trajectory_ids = unique_trajectory_ids[-batch_size:]

        # Lọc các sample dựa trên trajectory id được chọn
        selected_indices = [
            i for i, trajectory_id in enumerate(buffer_large["trajectory_ids"])
            if trajectory_id in selected_trajectory_ids
        ]

        # Shuffle để đảm bảo tính ngẫu nhiên
        np.random.shuffle(selected_indices)

        # Trả về batch dữ liệu
        batch_dict = {
            "states": np.array([buffer_large["states"][i] for i in selected_indices]),  # 2D
            "actions": np.array([buffer_large["actions"][i] for i in selected_indices]),  # 2D
            "log_probs": np.array([buffer_large["log_probs"][i] for i in selected_indices]).reshape(-1, 1),  # 1D -> 2D
            "sigma": np.array([buffer_large["sigma"][i] for i in selected_indices]),  # 2D
            "returns": np.array([buffer_large["returns"][i] for i in selected_indices]).reshape(-1, 1),  # 1D -> 2D
            "advantages": np.array([buffer_large["advantages"][i] for i in selected_indices]).reshape(-1, 1),  # 1D -> 2D
            "trajectory_ids": np.array([buffer_large["trajectory_ids"][i] for i in selected_indices]),
        }

        # ================== TẠO ĐẦU VÀO CHO VIỆC HUẤN LUYỆN ===============
        """
        ĐẦU RA LÀ CÁC TENSOR CÓ SHAPE LÀ : [batchsize, num_samples, feature]
        """
        # 1. Lấy danh sách các giá trị duy nhất trong "trajectory_ids"
        unique_trajectory_ids = np.unique(buffer_large["trajectory_ids"])

        # 2. Khởi tạo dictionary kết quả
        result = {}

        # Duyệt qua tất cả các keys và chuyển đổi dữ liệu thành tensor
        trajectory_ids = batch_dict["trajectory_ids"]  # Dữ liệu trajectory_ids
        for key in batch_dict.keys():
            if key != "trajectory_ids":
                data = batch_dict[key]  # Dữ liệu cho key hiện tại

                # Tạo một mảng chứa mask cho mỗi trajectory_id
                mask_list = [trajectory_ids == trajectory_id for trajectory_id in unique_trajectory_ids]

                # Dùng broadcasting và slicing để nhóm dữ liệu theo trajectory_id
                grouped_data = [data[mask] for mask in mask_list]

                # Tìm số mẫu lớn nhất trong mỗi nhóm để padding
                max_samples = max(len(group) for group in grouped_data)

                # # Kiểm tra kích thước của từng group trước khi padding
                # print(f"Max samples: {max_samples}")
                # print(f"Grouped data shapes before padding: {[group.shape for group in grouped_data]}")

                # Padding dữ liệu sao cho tất cả các nhóm đều có số lượng mẫu như nhau
                padded_data = np.array(
                    [np.pad(group, ((0, max_samples - len(group)), (0, 0)), mode='constant') for group in grouped_data])

                # Chuyển thành tensor
                result[key] = torch.tensor(padded_data)

        return result

    def reset(self):
        """Reset buffer nhỏ trong RAM."""
        for key in self.buffer:
            self.buffer[key] = deque(maxlen=self.trajectory_size)


# ========= REPLAY CÓ KÍCH THƯỚC NHỎ ĐỂ CHẠY ĐA LUỒNG =============
class ReplayCache(Buffer):
    def __init__(self, trajectory_size=4 * 300, max_size=6.250):
        # Gọi hàm __init__ của class Buffer để khởi tạo buffer
        super().__init__(trajectory_size, max_size)




