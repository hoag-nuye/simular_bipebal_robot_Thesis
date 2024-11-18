import numpy as np
from typing import List, Dict, Tuple
import support as sp

class Trajectory:
    def __init__(self, trajectory_id: int):
        """
        Khởi tạo một đối tượng trajectory.

        Args:
            trajectory_id (int): ID của trajectory này.
        """
        self.trajectory_id = trajectory_id  # ID duy nhất cho trajectory
        self.states: List[Dict] = []  # Danh sách chứa các trạng thái (state)

    def add_state(self,
                  time_step: int,
                  pelvis_orientation: np.ndarray,
                  rotation_velocity: np.ndarray,
                  joint_positions: np.ndarray,
                  joint_velocities: np.ndarray,
                  X_des: float,
                  Y_des: float,
                  r: float,
                  p: float,
                  theta: float,
                  delta_phase: float,
                  At: float,
                  Gt: float):
        """
        Thêm một trạng thái (state) vào trajectory.

        Args:
            time_step (int): Bước thời gian.
            pelvis_orientation (np.ndarray): Hướng xương chậu (vector 3D, đơn vị: quaternion hoặc Euler angles).
            rotation_velocity (np.ndarray): Vận tốc góc của xương chậu (vector 3D, đơn vị: rad/s).
            joint_positions (np.ndarray): Vị trí các khớp (vector 10D, đơn vị: rad).
            joint_velocities (np.ndarray): Vận tốc các khớp (vector 10D, đơn vị: rad/s).
            X_des (float): Vận tốc mong muốn theo trục X (m/s).
            Y_des (float): Vận tốc mong muốn theo trục Y (m/s).
            r (float): Tỷ lệ thời gian giữa swing và cycle.
            p (float): Tín hiệu clock (clock signal).
            theta (float): Pha của chu kỳ.
            delta_phase (float): Độ lệch pha giữa chu kỳ hiện tại và lý tưởng.
            At (float): Hành động tại thời điểm (action từ policy).
            Gt (float): Phần thưởng tại thời điểm (reward).
        """

        # Validate tham số đầu vào
        sp.validate_add_state(pelvis_orientation, (3,), "pelvis_orientation")
        sp.validate_add_state(rotation_velocity, (3,), "rotation_velocity")
        sp.validate_add_state(joint_positions, (10,), "joint_positions")
        sp.validate_add_state(joint_velocities, (10,), "joint_velocities")

        # Nếu tham số đầu vào hợp lệ thì thêm thành công
        state = {
            "time_step": time_step,
            "pelvis_orientation": pelvis_orientation,  # (3D vector)
            "rotation_velocity": rotation_velocity,  # (3D vector)
            "joint_positions": joint_positions,  # (10D vector)
            "joint_velocities": joint_velocities,  # (10D vector)
            "X_des": X_des,  # (float)
            "Y_des": Y_des,  # (float)
            "r": r,  # (float)
            "p": p,  # (float)
            "theta": theta,  # (float)
            "delta_phase": delta_phase,  # (float)
            "At": At,  # (float)
            "Gt": Gt  # (float)
        }
        self.states.append(state)

    def to_dict(self) -> Dict:
        """
        Chuyển toàn bộ trajectory thành một dictionary.

        Returns:
            Dict: Dữ liệu trajectory bao gồm ID và các trạng thái.
        """
        return {
            "trajectory_id": self.trajectory_id,
            "states": self.states
        }
