# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass


# =================== Giá trị của trạng thái S_t trong xml =====================
class StateFields(Enum):
    # <!-- Joint positions -->
    joint_positions = ["left-hip-roll-output",
                       "left-hip-yaw-output",
                       "left-hip-pitch-output",
                       "left-knee-output",
                       "left-foot-output",
                       "left-shin-output",
                       "left-tarsus-output",

                       "right-hip-roll-output",
                       "right-hip-yaw-output",
                       "right-hip-pitch-output",
                       "right-knee-output",
                       "right-foot-output",
                       "right-shin-output",
                       "right-tarsus-output",
                       ]

    # <!-- Joint velocities -->
    joint_velocities = ["left-hip-roll-velocity",
                        "left-hip-yaw-velocity",
                        "left-hip-pitch-velocity",
                        "left-knee-velocity",
                        "left-foot-velocity",
                        "left-shin-velocity",
                        "left-tarsus-velocity",

                        "right-hip-roll-velocity",
                        "right-hip-yaw-velocity",
                        "right-hip-pitch-velocity",
                        "right-knee-velocity",
                        "right-foot-velocity",
                        "right-shin-velocity",
                        "right-tarsus-velocity"]

    # <!-- Touch sensors -->
    left_foot_touch = ["left-foot-touch"]
    right_foot_touch = ["right-foot-touch"]

    # <!-- Force sensors -->
    left_foot_force = ["left-foot-force"]
    right_foot_force = ["right-foot-force"]

    # <!-- Speed sensors -->
    left_foot_speed = ["left-foot-speed"]
    right_foot_speed = ["right-foot-speed"]

    # <!-- Pelvis sensors -->
    pelvis_orientation = ["pelvis-orientation"]  # Đo lường hướng của xương chậu trong không gian
    pelvis_velocity = ["pelvis-velocity"]  # Vận tốc của xương chậu
    pelvis_angular_velocity = ["pelvis-angular-velocity"]  # Vận tốc xoay của xương chậu
    pelvis_linear_acceleration = ["pelvis-linear-acceleration"]  # Gia tốc tuyến tính của xương chậu


# ================== Giá trị của các khớp ==============
class JointFields(Enum):
    joint_positions = ["left-hip-roll",
                       "left-hip-yaw",
                       "left-hip-pitch",
                       "left-knee",
                       "left-foot",

                       "right-hip-roll",
                       "right-hip-yaw",
                       "right-hip-pitch",
                       "right-knee",
                       "right-foot"]


# =================== Giá trị của actuator trong xml =====================

# ĐÂY LÀ CHỨA VỊ TRÍ CỦA CÁC KHỚP CÓ ACTUATOR ĐIỀU KHIỂN ()
class ActuatorFields(Enum):
    # <!-- Actuator positions -->
    actuator_positions = ["left-hip-roll-output",
                          "left-hip-yaw-output",
                          "left-hip-pitch-output",
                          "left-knee-output",
                          "left-foot-output",

                          "right-hip-roll-output",
                          "right-hip-yaw-output",
                          "right-hip-pitch-output",
                          "right-knee-output",
                          "right-foot-output",
                          ]

    # <!-- Actuator velocities -->
    actuator_velocities = ["left-hip-roll-velocity",
                           "left-hip-yaw-velocity",
                           "left-hip-pitch-velocity",
                           "left-knee-velocity",
                           "left-foot-velocity",

                           "right-hip-roll-velocity",
                           "right-hip-yaw-velocity",
                           "right-hip-pitch-velocity",
                           "right-knee-velocity",
                           "right-foot-velocity", ]


@dataclass
class AgentState:
    time_step: int
    isTerminalState: bool  # Kiểm tra xem có phải trạng thái cuối hay không
    joint_positions: np.ndarray  # (6D vector) Vị trí góc quay của khớp
    joint_velocities: np.ndarray  # (6D vector) Vận tốc góc của khớp
    left_foot_touch: np.ndarray  # (1D vector)  Tiếp xúc chân trái của pelvis
    right_foot_touch: np.ndarray  # (1D vector) Tiếp xúc chân phải
    left_foot_force: np.ndarray  # (3D vector) Lực tiếp xúc chân trái của pelvis
    right_foot_force: np.ndarray  # (3D vector) Lực tiếp xúc chân phải
    left_foot_speed: np.ndarray  # (3D vector) Vận tốc chân trái của pelvis
    right_foot_speed: np.ndarray  # (3D vector) Vận tốc chân phải của pelvis
    pelvis_orientation: np.ndarray  # (4D vector) Hướng xoay của pelvis
    pelvis_velocity: np.ndarray  # (3D vector) Vận tốc của pelvis
    pelvis_angular_velocity: np.ndarray  # (3D vector) Vận tốc xoay của pelvis
    pelvis_linear_acceleration: np.ndarray  # (3D vector) Gia tốc của pelvis

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def as_tensor_list(self):
        # Convert tất cả các giá trị cần thiết thành torch tensor
        # Bỏ qua time_step vì đây không phải là array mà là một số nguyên
        return [
            torch.tensor(self.joint_positions),
            torch.tensor(self.joint_velocities),
            torch.tensor(self.left_foot_force),
            torch.tensor(self.right_foot_force),
            torch.tensor(self.left_foot_speed),
            torch.tensor(self.right_foot_speed),
            torch.tensor(self.pelvis_orientation),
            torch.tensor(self.pelvis_velocity),
            torch.tensor(self.pelvis_angular_velocity),
            torch.tensor(self.pelvis_linear_acceleration)
        ]


# =================== Tham số đầu vào để tính reward =====================
@dataclass
class RewardParam:
    S_t: AgentState
    torque_t: np.ndarray
    action_t: np.ndarray
    action_t_sub1: np.ndarray
    r_swing: float
    tms_clk: int
    ECfrc_left: float
    ECfrc_right: float
    ECspd_left: float
    ECspd_right: float
    x_des: float
    y_des: float
    quat_des: np.ndarray
    fall_reward : float


@dataclass
class OutputActorModel:
    pTarget_mu: torch.Tensor
    dTarget_mu: torch.Tensor
    pGain_mu: torch.Tensor
    dGain_mu: torch.Tensor
    sigma: torch.Tensor
