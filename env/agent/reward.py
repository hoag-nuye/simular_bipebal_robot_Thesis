# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import math
import numpy as np
from matplotlib import pyplot as plt

from env.agent.dataclass_agt import RewardParam


class RewardsHistory:
    def __init__(self):
        self.buffer = {
            "id_r": [],
            "R_multi": [],
            "r_bipedal": [],
            "r_cmd": [],
            "r_smooth": [],
            "r_std_cost": [],
        }

    def add_reward(self, R_multi, r_bipedal, r_cmd, r_smooth, r_std_cost, id_r):
        print("dã thêm ", id_r)
        """Thêm một mẫu vào buffer nhỏ (RAM)."""
        self.buffer["id_r"].append(id_r)
        self.buffer["R_multi"].append(R_multi)
        self.buffer["r_bipedal"].append(r_bipedal)
        self.buffer["r_cmd"].append(r_cmd)
        self.buffer["r_smooth"].append(r_smooth)
        self.buffer["r_std_cost"].append(r_std_cost)

    def plot_rewards(self):
        """Vẽ 5 biểu đồ trong cùng một khung hiển thị."""

        ids = self.buffer["id_r"]
        print(len(ids))

        plt.figure(figsize=(12, 8))

        plt.plot(ids, self.buffer["R_multi"], label="R_multi", marker='o')
        plt.plot(ids, self.buffer["r_bipedal"], label="r_bipedal", marker='x')
        plt.plot(ids, self.buffer["r_cmd"], label="r_cmd", marker='s')
        plt.plot(ids, self.buffer["r_smooth"], label="r_smooth", marker='^')
        plt.plot(ids, self.buffer["r_std_cost"], label="r_std_cost", marker='d')

        plt.xlabel("ID")
        plt.ylabel("Reward Values")
        plt.title("Reward Components vs ID")
        plt.legend()
        plt.grid(True)
        plt.show()


def compute_reward_2th(param: RewardParam):
    S_t = param.S_t
    action_t = param.action_t
    action_t_sub1 = param.action_t_sub1
    r_swing = param.r_swing
    tms_clk = param.tms_clk
    ECfrc_left = param.ECfrc_left[tms_clk]
    ECfrc_right = param.ECfrc_right[tms_clk]
    ECspd_left = param.ECspd_left[tms_clk]
    ECspd_right = param.ECspd_right[tms_clk]
    x_des = param.x_des
    y_des = param.y_des
    quat_des = param.quat_des

    R_multi = 0
    return R_multi


# ĐƯA vận tốc của imu về đúng vận tốc song song với mặt sàn

def compute_reward(param: RewardParam):
    S_t = param.S_t
    torque = param.torque_t
    action_t = param.action_t
    action_t_sub1 = param.action_t_sub1
    r_swing = param.r_swing
    tms_clk = param.tms_clk
    ECfrc_left = param.ECfrc_left[tms_clk]
    ECfrc_right = param.ECfrc_right[tms_clk]
    ECspd_left = param.ECspd_left[tms_clk]
    ECspd_right = param.ECspd_right[tms_clk]
    x_des = param.x_des
    y_des = param.y_des
    quat_des = param.quat_des
    fall_reward = param.fall_reward
    # =========== Hệ số Beta =============
    beta = 1  # Tránh việc phần thưởng quá nhỏ dẫn đến sự biến mất của đạo hàm
    # =========== Hệ số omega =============
    omega = (1 + math.exp(-50 * (r_swing - 0.15))) ** (-1)
    # =========== Tính R of Bipedal ==========

    left_foot_force = np.linalg.norm(S_t.left_foot_force) ** 2 / 100
    right_foot_force = np.linalg.norm(S_t.right_foot_force) ** 2 / 100
    left_foot_speed = np.linalg.norm(S_t.left_foot_speed) ** 2
    right_foot_speed = np.linalg.norm(S_t.right_foot_speed) ** 2
    # print(S_t.left_foot_touch, S_t.right_foot_touch)
    q_left_frc = 1 - math.exp(-omega * left_foot_force)
    q_right_frc = 1 - math.exp(-omega * right_foot_force)
    q_left_spd = 1 - math.exp(-2 * omega * left_foot_speed)
    q_right_spd = 1 - math.exp(-2 * omega * right_foot_speed)
    # print(f"R Force chân trái: {q_left_frc}")
    # print(f"R Force chân phải: {q_right_frc}")
    # print(f"R Speed chân trái: {q_left_spd}")
    # print(f"R Speed chân phải: {q_right_spd}")

    r_bipedal = ECfrc_left * q_left_frc \
                + ECfrc_right * q_right_frc \
                + ECspd_left * q_left_spd \
                + ECspd_right * q_right_spd
    # =========== Tính R of Cmd ==========
    q_x = 1 - math.exp(-2 * omega * abs(x_des - S_t.pelvis_velocity[0]))  # x_des = vận tốc mục tiêu theo trục x
    q_y = 1 - math.exp(-2 * omega * abs(y_des + S_t.pelvis_velocity[1]))
    q_orientation = 1 - math.exp(-3 * (1 - np.dot(S_t.pelvis_orientation, quat_des) ** 2))  # quaternion

    r_cmd = (-1) * q_x \
            + (-1) * q_y \
            + (-1) * q_orientation
    # =========== Tính R of Smooth ==========
    # Vận tốc góc và gia tốc
    q_action_diff = 1 - math.exp(-5 * np.linalg.norm(action_t - action_t_sub1))
    q_torque = 1 - math.exp(-0.05 * np.linalg.norm(torque))
    q_pelvis_acc = 1 - math.exp(-0.10 * (np.linalg.norm(S_t.pelvis_angular_velocity)
                                         + np.linalg.norm([S_t.pelvis_linear_acceleration])))

    r_smooth = (-1) * q_action_diff \
               + (-1) * q_torque \
               + (-1) * q_pelvis_acc
    # # =========== Tính R of Standing cost ==========

    # Tính err_sym : Symmetry Error
    err_sym = np.linalg.norm(S_t.left_foot_force - S_t.right_foot_force)

    q_std_cost = 1 - math.exp(-(err_sym + 20 * q_action_diff))

    r_std_cost = (omega - 1) * q_std_cost

    # ============== TÍNH R Multi ==============
    R_multi = 0.400 * r_bipedal \
              + 0.300 * r_cmd \
              + 0.100 * r_smooth \
              + 0.100 * r_std_cost \
              + beta
              # + fall_reward \


    # print(f"r_bipedal - Bước: {r_bipedal}")
    # print(f"r_cmd - Tránh ngã: {r_cmd}")
    # print(f"r_smooth - Di chuyển mượt: {r_smooth}")
    # print(f"r_std_cost - Đứng thẳng : {r_std_cost}")
    # print(f"R_multi - Tổng R: {R_multi}")
    # print("=======================================")
    return R_multi
