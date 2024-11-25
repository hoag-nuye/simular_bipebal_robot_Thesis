import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass


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

    # <!-- Pelvis sensors -->
    pelvis_orientation = ["pelvis-orientation"]  # Đo lường hướng của xương chậu trong không gian
    pelvis_linear_acceleration = ["pelvis-linear-acceleration"]  # Gia tốc tuyến tính của xương chậu
    pelvis_velocity = ["pelvis-velocity"]


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
    PELVIS_VELOCITY = "pelvis-velocity"


@dataclass
class AgentState:
    time_step: int
    joint_positions: np.ndarray  # (6D vector) Vị trí góc quay của khớp
    joint_velocities: np.ndarray  # (6D vector) Vận tốc góc của khớp
    left_foot_force: np.ndarray  # (0D vector) Lực tiếp xúc chân trái của pelvis : np.linalg.norm(3D vector)
    right_foot_force: np.ndarray  # (0D vector) Lực tiếp xúc chân phải : np.linalg.norm(3D vector)
    pelvis_orientation: np.ndarray  # (4D vector) Hướng xoay của pelvis
    pelvis_velocity: np.ndarray  # (3D vector) Vận tốc của pelvis
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
            torch.tensor(self.left_foot_force).unsqueeze(0),  # Chuyển thành vector 1D
            torch.tensor(self.right_foot_force).unsqueeze(0),  # Chuyển thành vector 1D
            torch.tensor(self.pelvis_orientation),
            torch.tensor(self.pelvis_velocity),
            torch.tensor(self.pelvis_linear_acceleration)
        ]
