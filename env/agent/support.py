# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import numpy as np
import math

from numpy import i0


def validate_add_state(array: np.ndarray, expected_shape: tuple, name: str):
    """
    Kiểm tra xem mảng đầu vào có đúng kích thước không.

    Args:
        array (np.ndarray): Mảng cần kiểm tra.
        expected_shape (tuple): Kích thước mong đợi (vd: (3,), (10,)).
        name (str): Tên của tham số để in thông báo lỗi nếu sai.

    Raises:
        ValueError: Nếu kích thước mảng không đúng.
    """
    if array.shape != expected_shape:
        raise ValueError(
            f"Tham số '{name}' có kích thước {array.shape}, nhưng kích thước mong đợi là {expected_shape}."
        )


# ================ Get clock p and E_C_frc, E_C_spd ===============

# Tính xác suất P(A_i < phi < B_i) với A_i và B_i theo phân phối Von Mises
def __von_mises_prob_quick_decay(phi, a_i, b_i, kappa, L):
    """
    Tính xác suất P(A_i < phi < B_i) với A_i và B_i theo phân phối Von Mises,
    có tính tuần hoàn với chu kỳ L.

    Tham số:
    - phi: giá trị cần kiểm tra, điều chỉnh theo chu kỳ L
    - a_i: trung bình của A_i (thời điểm bắt đầu) (cố định)
    - b_i: trung bình của B_i (thời điểm kết thúc) (cố định)
    - kappa: tham số tập trung của phân phối Von Mises
    - L: chu kỳ tuần hoàn

    Kết quả:
    - Xác suất phi nằm trong khoảng (A_i, B_i) theo phân phối Von Mises.
    """
    # Điều chỉnh phi, a_i, b_i về chu kỳ [0, L]
    phi = phi % L
    a_i = a_i % L
    b_i = b_i % L

    # Trung bình của phân phối là trung tâm của (a_i, b_i)
    if a_i <= b_i:
        mu = (a_i + b_i) / 2
    else:
        # Nếu khoảng vượt qua chu kỳ, điều chỉnh trung bình
        mu = ((a_i + b_i + L) / 2) % L

    # Tính khoảng cách tuần hoàn
    d = min(abs(phi - mu), abs(phi - mu + L), abs(phi - mu - L))

    # Tính xác suất theo phân phối Von Mises
    prob = np.exp(kappa * np.cos(2 * np.pi * d / L)) / (2 * np.pi * i0(kappa))

    return prob

# N = 100
# kappa = 20.0 # độ xoải của phân bố
# r = 0.55 # Tỉ lệ pha swing
# theta_left = 0.0 # tham số bù của chu kì
# theta_right = 0.5
# theta_phase_dif = abs(theta_left-theta_right) # độ lệch pha
# L = 1 # độ dài của 1 chu kì


def get_clock(r, theta_left, theta_right, a_i, N=100, kappa=20.0, L=1):

    theta_phase_dif = abs(theta_left - theta_right)  #tính độ lệch pha
    a_i = a_i  # tạo pha đầu của swing
    b_i = a_i + r*L  # tạo pha kết thúc của swing

    X_phi = np.linspace(0, L, N)  # Tạo N bước(pha) trong 1 chu kì

    C_frc_left = np.array([__von_mises_prob_quick_decay(phi, a_i, b_i, kappa, L) for phi in X_phi])  # tạo giá trị C_frc cho chân trái
    C_frc_left[C_frc_left < 1e-2] = 0  # cắt giá trị quá nhỏ và cho bằng 0

    C_frc_left = C_frc_left - 1
    C_frc_right = np.array([__von_mises_prob_quick_decay(phi, a_i+theta_phase_dif, b_i+theta_phase_dif, kappa, L) for phi in X_phi]) - 1  # tạo giá trị C_frc cho chân phải với 1 độ lệch pha

    # C_spd_left đối xứng với C_frc_right qua trục -0.5 vì ta sẽ phạt lực ở pha swing và phạt vận tốc ở pha stance
    # Tìm đường đối xứng : d(A, y=−0.5)=−d(B, y=−0.5) -> y' - (-0.5) = -(y - (-0.5)) -> y' = -1 - y
    C_spd_left = -1 - C_frc_left
    C_spd_right = -1 - C_frc_right

    # Lưu giá trị vừa tạo
    ECfrc_left = C_frc_left
    ECfrc_right = C_frc_right
    ECspd_left = C_spd_left
    ECspd_right = C_spd_right

    p = [[math.sin(2*math.pi*(phi+theta_left)/N), math.sin(2*math.pi*(phi+theta_right)/N)] for phi in X_phi]

    return p, ECfrc_left, ECfrc_right, ECspd_left, ECspd_right

# # Ví dụ dictionary chứa các tensor
# tensor_dict = {
#     'tensor1': torch.randn(32, 10),  # Tensor có kích thước [32, 10]
#     'tensor2': torch.randn(32, 20),  # Tensor có kích thước [32, 20]
# }
#
# # Mảng chứa các chỉ số ngẫu nhiên
# indices = torch.tensor([2, 7, 12, 25, 30])  # Ví dụ mảng chỉ số
#
# # Tạo dictionary mới chứa các tensor đã lọc
# filtered_dict = {key: tensor[indices] for key, tensor in tensor_dict.items()}
#
# # Kiểm tra kết quả
# for key, tensor in filtered_dict.items():
#     print(f"{key}: {tensor.shape}")