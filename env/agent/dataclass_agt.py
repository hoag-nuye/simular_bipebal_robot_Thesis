from dataclasses import dataclass
import numpy as np

from enum import Enum

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
    PELVIS_ORIENTATION = "pelvis-orientation"  # (Trùng với trên) Đo lường hướng của xương chậu trong không gian
    PELVIS_ANGULAR_VELOCITY = "pelvis-angular-velocity"  # Vận tốc góc của xương chậu
    PELVIS_LINEAR_ACCELERATION = "pelvis-linear-acceleration"  # Gia tốc tuyến tính của xương chậu
    PELVIS_MAGNETOMETER = "pelvis-magnetometer"  # Cảm biến từ trường đo trạng thái của xương chậu so với từ trường Trái đất


@dataclass
class AgentState:
    time_step: int
    pelvis_orientation: np.ndarray  # (3D vector)
    rotation_velocity: np.ndarray  # (3D vector)
    joint_positions: np.ndarray  # (10D vector)
    joint_velocities: np.ndarray  # (10D vector)
    X_des: float
    Y_des: float
    r: float
    p: float
    theta: float
    delta_phase: float
    At: float
    Gt: float
