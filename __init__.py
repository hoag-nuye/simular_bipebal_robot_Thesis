# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import os
import multiprocessing

import torch

from env.agent.mujoco_agt import Agent
from env.mujoco_env import Environment
from interface.mujoco_viewer import (mujoco_viewer_process,
                                     mujoco_viewer_process_begin,
                                     mujoco_viewer_process_external_state)

from interface.plot_param import plot_param_process

# Số lõi logic (logical cores)
logical_cores = os.cpu_count()
print(f"Số logical cores: {logical_cores}")

# Số lõi vật lý (physical cores)
physical_cores = multiprocessing.cpu_count()
print(f"Số physical cores: {physical_cores}")

print(multiprocessing)

# -------------- initial environment --------------------
agt_xml_path = 'structures/agility_cassie/cassie.xml'
agt = Agent(agt_xml_path)

# -------------- Add a new agent ---------------------
env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
agt.add_env(env, 'cassie_env')
agt.set_state(0)  # Khởi tạo giá trị ban đầu

# mujoco_viewer_process_begin(agt)
plot_param_process("models/param/")
# mujoco_viewer_process(agt=agt, policy_freq=40, pd_control_freq=2000, use_cuda=True)
# mujoco_viewer_process_external_state(agt=agt, policy_freq=40, pd_control_freq=2000, use_cuda=True)



# ==================== TEST MAX REWARD =============================
# import math
# import numpy as np
# def compute_reward():
#     ECfrc_left = 0
#     ECfrc_right = -1
#     ECspd_left = -1
#     ECspd_right = 0
#     x_des = 0
#     y_des = 0
#     quat_des = [1,0,0,0]
#     # =========== Hệ số Beta =============
#     beta = 1  # Tránh việc phần thưởng quá nhỏ dẫn đến sự biến mất của đạo hàm
#     # =========== Hệ số omega =============
#     omega = (1 + math.exp(-50 * (0.6 - 0.15))) ** (-1)
#     # =========== Tính R of Bipedal ==========
#     q_left_frc = 1 - math.exp(-omega * np.linalg.norm(100) ** 2 / 100)
#     q_right_frc = 1 - math.exp(-omega * np.linalg.norm(0) ** 2 / 100)
#     q_left_spd = 1 - math.exp(-2 * omega * np.linalg.norm(0) ** 2)
#     q_right_spd = 1 - math.exp(-2 * omega * np.linalg.norm(10) ** 2)
#
#     r_bipedal = ECfrc_left * q_left_frc \
#               + ECfrc_right * q_right_frc \
#               + ECspd_left * q_left_spd \
#               + ECspd_right * q_right_spd
#     # =========== Tính R of Cmd ==========
#     q_x = 1 - math.exp(-2 * omega * abs(x_des - 0))
#     q_y = 1 - math.exp(-2 * omega * abs(y_des - 0))
#     q_orientation = 1 - math.exp(-3 * (1 - np.dot([1,0,0,0], quat_des) ** 2))  # quaternion
#
#     r_cmd = (-1) * q_x \
#           + (-1) * q_y \
#           + (-1) * q_orientation
#     # =========== Tính R of Smooth ==========
#     q_action_diff = 1 - math.exp(-5 * np.linalg.norm(0))
#     q_torque = 1 - math.exp(-0.05 * np.linalg.norm(0))
#     q_pelvis_acc = 1 - math.exp(-0.10 * (np.linalg.norm([0,0,0])
#                                          + np.linalg.norm([0,0,0])))
#
#     r_smooth = (-1) * q_action_diff \
#              + (-1) * q_torque \
#              + (-1) * q_pelvis_acc
#     # =========== Tính R of Standing cost ==========
#     w = 1
#     x = 0
#     y = 0
#     z = 0
#
#     # Tính Roll
#     roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
#     # Tính Pitch
#     pitch = np.arcsin(2 * (w * y - z * x))
#     # Tính err_sym : Symmetry Error
#     err_sym = abs(roll) + abs(pitch)
#
#     q_std_cost = 1 - math.exp(-(err_sym + 20 * q_action_diff))
#
#     r_std_cost = (omega - 1) * q_std_cost
#
#     # ============== TÍNH R Multi ==============
#     R_multi = 0.400 * r_bipedal \
#             + 0.300 * r_cmd \
#             + 0.100 * r_smooth \
#             + 0.100 * r_std_cost \
#             + beta
#     return R_multi
#
# print(f"Reward: {compute_reward()}")