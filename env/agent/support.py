import numpy as np
import math

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
    có tính tuần hoàn với chu kỳ L và lưu khoảng thừa để gán vào đầu chu kỳ.

    Tham số:
    - phi: giá trị cần kiểm tra, điều chỉnh theo chu kỳ L
    - a_i: trung bình của A_i (thời điểm bắt đầu) (cố định)
    - b_i: trung bình của B_i (thời điểm kết thúc) (cố định)
    - kappa: tham số tập trung của phân phối Von Mises
    - L: chu kỳ tuần hoàn

    Kết quả:
    - Xác suất phi nằm trong khoảng (A_i, B_i) có tính tuần hoàn và mịn với khoảng thừa.
    """
    # Điều chỉnh phi về chu kỳ [0, L] mà không thay đổi a_i và b_i
    phi = phi % L
    a_i = a_i % L
    b_i = b_i % L

    # Lưu phần thừa khi vượt ngoài chu kỳ
    # Trường hợp không vượt qua điểm L
    if a_i < b_i:
        if a_i <= phi <= b_i:
            return 1.0
        else:
            # Tính khoảng cách vòng qua biên L
            d = min(abs(phi - a_i), abs(phi - b_i),
                    abs(phi - a_i + L), abs(phi - b_i + L))
            return np.exp(- (kappa * d) ** 2)
    else:
        # Trường hợp vượt qua điểm L, chia khoảng làm hai
        if phi >= a_i or phi <= b_i:
            return 1.0
        else:
            # Khoảng cách vòng qua biên, bao gồm phần thừa
            d = min(abs(phi - a_i), abs(phi - b_i),
                    abs(phi - a_i + L), abs(phi - b_i + L))
            return np.exp(- (kappa * d) ** 2)

# N = 100
# kappa = 20.0 # độ xoải của phân bố
# r = 0.55 # Tỉ lệ pha swing
# theta_left = 0.0 # tham số bù của chu kì
# theta_right = 0.5
# theta_phase_dif = abs(theta_left-theta_right) # độ lệch pha
# L = 1 # độ dài của 1 chu kì


def get_clock(r, theta_left, theta_right, N=100, kappa=20.0, L=1):

    theta_phase_dif = abs(theta_left - theta_right)  #tính độ lệch pha
    a_i = np.random.uniform(low=0.0, high=1-r*L)  # tạo pha đầu của swing
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

