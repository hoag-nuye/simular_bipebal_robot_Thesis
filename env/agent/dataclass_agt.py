from dataclasses import dataclass
import numpy as np

from enum import Enum


class SensorFields(Enum):

    LEFT_HIP_ROLL_INPUT = "left_hip_roll_input"  # Đầu vào điều khiển cho khớp lăn hông trái (lăn: roll)
    LEFT_HIP_YAW_INPUT = "left_hip_yaw_input"  # Đầu vào điều khiển cho khớp xoay hông trái (xoay: yaw)
    LEFT_HIP_PITCH_INPUT = "left_hip_pitch_input"  # Đầu vào điều khiển cho khớp gập hông trái (gập: pitch)
    LEFT_KNEE_INPUT = "left_knee_input"  # Đầu vào điều khiển cho khớp gối trái
    LEFT_FOOT_INPUT = "left_foot_input"  # Đầu vào điều khiển cho bàn chân trái

    RIGHT_HIP_ROLL_INPUT = "right_hip_roll_input"  # Đầu vào điều khiển cho khớp lăn hông phải
    RIGHT_HIP_YAW_INPUT = "right_hip_yaw_input"  # Đầu vào điều khiển cho khớp xoay hông phải
    RIGHT_HIP_PITCH_INPUT = "right_hip_pitch_input"  # Đầu vào điều khiển cho khớp gập hông phải
    RIGHT_KNEE_INPUT = "right_knee_input"  # Đầu vào điều khiển cho khớp gối phải
    RIGHT_FOOT_INPUT = "right_foot_input"  # Đầu vào điều khiển cho bàn chân phải

    LEFT_SHIN_OUTPUT = "left_shin_output"  # Đầu ra từ cảm biến đo lường trạng thái của ống chân trái
    LEFT_TARSUS_OUTPUT = "left_tarsus_output"  # Đầu ra từ cảm biến đo trạng thái của khớp cổ chân trái
    LEFT_FOOT_OUTPUT = "left_foot_output"  # Đầu ra từ cảm biến đo lường trạng thái bàn chân trái

    RIGHT_SHIN_OUTPUT = "right_shin_output"  # Đầu ra từ cảm biến đo trạng thái của ống chân phải
    RIGHT_TARSUS_OUTPUT = "right_tarsus_output"  # Đầu ra từ cảm biến đo trạng thái của khớp cổ chân phải
    RIGHT_FOOT_OUTPUT = "right_foot_output"  # Đầu ra từ cảm biến đo lường trạng thái bàn chân phải

    PELVIS_ORIENTATION = "pelvis_orientation"  # (Trùng với trên) Đo lường hướng của xương chậu trong không gian
    PELVIS_ANGULAR_VELOCITY = "pelvis_angular_velocity"  # Vận tốc góc của xương chậu
    PELVIS_LINEAR_ACCELERATION = "pelvis_linear_acceleration"  # Gia tốc tuyến tính của xương chậu
    PELVIS_MAGNETOMETER = "pelvis_magnetometer"  # Cảm biến từ trường đo trạng thái của xương chậu so với từ trường Trái đất


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
