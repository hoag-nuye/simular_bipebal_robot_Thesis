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
    joint_positions = ["left-shin-output",
                       "left-tarsus-output",
                       "left-foot-output",
                       "right-shin-output",
                       "right-tarsus-output",
                       "right-foot-output"]

    # <!-- Joint velocities -->
    joint_velocities = ["left-shin-velocity",
                        "left-tarsus-velocity",
                        "left-foot-velocity",
                        "right-shin-velocity",
                        "right-tarsus-velocity",
                        "right-foot-velocity"]

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


# =================== Giá trị của actuator trong xml =====================
class ActuatorFields(Enum):
    # <!-- Actuator positions -->
    actuator_positions = ["left-hip-roll-input",
                          "left-hip-yaw-input",
                          "left-hip-pitch-input",
                          "left-knee-input",
                          "left-foot-input",

                          "right-hip-roll-input",
                          "right-hip-yaw-input",
                          "right-hip-pitch-input",
                          "right-knee-input",
                          "right-foot-input"]

    # <!-- Actuator velocities -->
    actuator_velocities = ["left-hip-roll-velocity",
                           "left-hip-yaw-velocity",
                           "left-hip-pitch-velocity",
                           "left-knee-velocity",
                           "left-foot-act-velocity",

                           "right-hip-roll-velocity",
                           "right-hip-yaw-velocity",
                           "right-hip-pitch-velocity",
                           "right-knee-velocity",
                           "right-foot-act-velocity"
                           ]


# =================== Giá trị của sensor trong xml =====================
class SensorFields(Enum):
    # <!-- Actuator positions -->
    LEFT_HIP_ROLL_INPUT = "left-hip-roll-input"  # Đầu vào điều khiển cho khớp lăn hông trái (lăn: roll)
    LEFT_HIP_YAW_INPUT = "left-hip-yaw-input"  # Đầu vào điều khiển cho khớp xoay hông trái (xoay: yaw)
    LEFT_HIP_PITCH_INPUT = "left-hip-pitch-input"  # Đầu vào điều khiển cho khớp gập hông trái (gập: pitch)
    LEFT_KNEE_INPUT = "left-knee-input"  # Đầu vào điều khiển cho khớp gối trái
    LEFT_FOOT_INPUT = "left-foot-input"  # Đầu vào điều khiển cho bàn chân trái

    RIGHT_HIP_ROLL_INPUT = "right-hip-roll-input"  # Đầu vào điều khiển cho khớp lăn hông phải
    RIGHT_HIP_YAW_INPUT = "right-hip-yaw-input"  # Đầu vào điều khiển cho khớp xoay hông phải
    RIGHT_HIP_PITCH_INPUT = "right-hip-pitch-input"  # Đầu vào điều khiển cho khớp gập hông phải
    RIGHT_KNEE_INPUT = "right-knee-input"  # Đầu vào điều khiển cho khớp gối phải
    RIGHT_FOOT_INPUT = "right-foot-input"  # Đầu vào điều khiển cho bàn chân phải

    # <!-- Actuator velocities -->
    LEFT_HIP_ROLL_VELOCITY = "left-hip-roll-velocity"  # Vận tốc khớp lăn hông trái
    LEFT_HIP_YAW_VELOCITY = "left-hip-yaw-velocity"  # Vận tốc khớp xoay hông trái
    LEFT_HIP_PITCH_VELOCITY = "left-hip-pitch-velocity"  # Vận tốc khớp gập hông trái
    LEFT_KNEE_VELOCITY = "left-knee-velocity"  # Vận tốc khớp gối trái
    LEFT_FOOT_ACT_VELOCITY = "left-foot-act-velocity"  # Vận tốc khớp bàn chân trái

    RIGHT_HIP_ROLL_VELOCITY = "right-hip-roll-velocity"  # Vận tốc khớp lăn hông phải
    RIGHT_HIP_YAW_VELOCITY = "right-hip-yaw-velocity"  # Vận tốc khớp xoay hông phải
    RIGHT_HIP_PITCH_VELOCITY = "right-hip-pitch-velocity"  # Vận tốc khớp gập hông phải
    RIGHT_KNEE_VELOCITY = "right-knee-velocity"  # Vận tốc khớp gối phải
    RIGHT_FOOT_ACT_VELOCITY = "right-foot-act-velocity"  # Vận tốc khớp bàn chân phải

    # <!-- Joint positions -->
    LEFT_SHIN_OUTPUT = "left-shin-output"  # Đầu ra từ cảm biến đo lường trạng thái của ống chân trái
    LEFT_TARSUS_OUTPUT = "left-tarsus-output"  # Đầu ra từ cảm biến đo trạng thái của khớp cổ chân trái
    LEFT_FOOT_OUTPUT = "left-foot-output"  # Đầu ra từ cảm biến đo lường trạng thái bàn chân trái
    RIGHT_SHIN_OUTPUT = "right-shin-output"  # Đầu ra từ cảm biến đo trạng thái của ống chân phải
    RIGHT_TARSUS_OUTPUT = "right-tarsus-output"  # Đầu ra từ cảm biến đo trạng thái của khớp cổ chân phải
    RIGHT_FOOT_OUTPUT = "right-foot-output"  # Đầu ra từ cảm biến đo lường trạng thái bàn chân phải

    # <!-- Joint velocities -->
    LEFT_SHIN_VELOCITY = "left-shin-velocity"
    LEFT_TARSUS_VELOCITY = "left-tarsus-velocity"
    LEFT_FOOT_VELOCITY = "left-foot-velocity"
    RIGHT_SHIN_VELOCITY = "right-shin-velocity"
    RIGHT_TARSUS_VELOCITY = "right-tarsus-velocity"
    RIGHT_FOOT_VELOCITY = "right-foot-velocity"

    # <!-- Force sensors -->
    LEFT_FOOT_FORCE = "left-foot-force"
    RIGHT_FOOT_FORCE = "right-foot-force"

    # <!-- Pelvis sensors -->
    PELVIS_ORIENTATION = "pelvis-orientation"  # Đo lường hướng của xương chậu trong không gian
    PELVIS_LINEAR_ACCELERATION = "pelvis-linear-acceleration"  # Gia tốc tuyến tính của xương chậu
    PELVIS_VELOCITY = "pelvis-velocity"  # Vận tốc của xương chậu
    PELVIS_ANGULAR_VELOCITY = "pelvis-angular-velocity"  # Vận tốc xoay của xương chậu


@dataclass
class AgentState:
    time_step: int
    isTerminalState: bool  # Kiểm tra xem có phải trạng thái cuối hay không
    joint_positions: np.ndarray  # (6D vector) Vị trí góc quay của khớp
    joint_velocities: np.ndarray  # (6D vector) Vận tốc góc của khớp
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

