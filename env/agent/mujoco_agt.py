# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import matplotlib.pyplot as plt
import math

import os
import time
import mujoco
import mujoco.viewer

import torch
import numpy as np

from lxml import etree
from env.mujoco_env import Environment

from env.agent.support import get_clock
from env.agent.dataclass_agt import AgentState, StateFields, ActuatorFields, JointFields

'''
# Mục lục các nhóm function
# ========================= Initial environment =========================
# ========================= Show simulation =========================
# ========================= Add a new env =========================
# ========================= OPERATION INFORMATION ======================
# ========================= STATE =========================
# ========================= MAPPING SENSOR ========================
# ========================= ACTION CONTROL =========================
# ========================= REWARD AND CLOCK =========================
# ========================= GET INPUT AND OUTPUT FOR MODEL =========================
'''


class Agent:

    # ========================= Initial environment =========================
    def __init__(self, agt_xml):

        self.agt_xml = agt_xml
        self.agt_model = mujoco.MjModel.from_xml_path(
            agt_xml)  # Provides the static structure of the model and is instantiated once from an XML file.
        self.agt_data = mujoco.MjData(self.agt_model)  # stores state and is updated throughout the simulation

        # save paused action of simulation
        self.paused = False

        # Create agent list
        self.envs = {}

        # Create map of variable of MjData
        self.sensors_map = {}
        self.sensors_data = []
        self.state_map = {}
        self.atr_map = {}
        self.atr_ctrl_map = {}

        # Save control signal
        self.p = None  # clock input
        self.r = None
        self.x_des_vel = None  # Control vận tốc theo trục x
        self.y_des_vel = None  # Control vận tốc theo trục y
        self.quat_des = None  # Control hướng

        # Save variable of reward (or different meaning is control signal)
        self.ECfrc_left = None
        self.ECfrc_right = None
        self.ECspd_left = None
        self.ECspd_right = None

        # Save info of present state
        self.S_t: AgentState = None  # Trạng thái S_t
        self.a_t: np.ndarray = None  # Hành động a_t
        self.V_t = None  # Ước tính giá trị S_t : V_t
        self.a_t_sub1 = []  # Hành động a_t-1
        self.R_t_add1 = None  # Phần thưởng R_t+1
        self.total_samples = 0

        # Save info of control
        self.atr_num = None
        self.atr_t = {}
        self.atr_ctrl_ranges = {}
        self.ctrl_min = None
        self.ctrl_max = None

        # Save Relay buffer and Trajectory set
        self.max_batch_size = None
        self.terminated_training = None
        self.max_relay_buffer_size = None
        self.relay_buffer = {}
        self.trajectory_set = {}

        # Setup for agent
        self.__setup()

    # Save important info of MjData
    def __setup(self):
        self.sensors_map = self.get_sensors_map()
        self.sensors_data = self.agt_data.sensordata
        self.state_map = self.__get_map_name2id(StateFields)
        self.atr_map = self.__get_map_name2id(
            ActuatorFields)  # Chứa đo vị trí sensor của khớp vị điều khiển bởi actutor
        self.atr_ctrl_map = self.get_actuators_map()  # Chứa vị trí của actuator
        self.atr_num = len(self.atr_map[next(iter(self.atr_map))])  # Tính số lượng actuator
        self.atr_ctrl_ranges = self.extract_ctrl_ranges()
        self.dTarget_ranges = self.extract_dTarget_ranges()
        self.a_t_sub1 = np.zeros(self.atr_num * 3)  # Được tính từ pTarget, pGain, dGain
        self.ctrl_min = torch.tensor([v[0] for v in self.atr_ctrl_ranges.values()])
        self.ctrl_max = torch.tensor([v[1] for v in self.atr_ctrl_ranges.values()])
        print('Setup is done !')

    # ========================= THIẾT LẬP TRẠNG THÁI BẮT ĐẦU =========================

    def get_foot_forces_z(self):
        left_foot_force = np.linalg.norm(self.S_t.left_foot_force) ** 2 / 100 * bool(self.S_t.left_foot_touch)
        right_foot_force = np.linalg.norm(self.S_t.right_foot_force) ** 2 / 100 * bool(self.S_t.right_foot_touch)
        return left_foot_force, right_foot_force

    def get_tilt_angle(self):
        w = self.S_t.pelvis_orientation[0].item()
        x = self.S_t.pelvis_orientation[1].item()
        y = self.S_t.pelvis_orientation[2].item()
        z = self.S_t.pelvis_orientation[3].item()
        """
            Chuyển đổi quaternion (x, y, z, w) sang góc Euler (roll, pitch, yaw).
            Các góc trả về theo đơn vị radian.
            """
        # Roll (góc quanh trục x)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        # Pitch (góc nghiêng quanh trục y)
        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))  # Giới hạn giá trị để tránh lỗi domain của arcsin
        pitch = math.asin(t2)

        # # Yaw (góc xoay quanh trục z)
        # t3 = 2.0 * (w * z + x * y)
        # t4 = 1.0 - 2.0 * (y * y + z * z)
        # yaw = math.atan2(t3, t4)

        return math.degrees(math.sqrt(roll ** 2 + pitch ** 2))

    def set_state_begin(self):
        # Đặt trạng thái ban đầu trong qpos (giá trị từ key XML)
        self.agt_data.qpos[:] = [0, 0, 1.0059301, 1, 0, 0, 0, 0.00449956, 0, 0.497301, 0.97861, -0.0164104,
                                 0.0177766, -0.204298, -1.1997, 0, 1.42671, -2.25907e-06, -1.52439, 1.50645,
                                 -1.59681, -0.00449956, 0, 0.497301, 0.97874, 0.0038687, -0.0151572, -0.204509,
                                 -1.1997, 0, 1.42671, 0, -1.52439, 1.50645, -1.59681
                                 ]

        # Cập nhật trạng thái mô phỏng
        mujoco.mj_forward(self.agt_model, self.agt_data)
        self.agt_data.qvel[:] = 0  # Đặt vận tốc bằng 0
        self.agt_data.ctrl[:] = 0  # Đặt lực điều khiển bằng 0

    # ========================= Show simulation =========================
    def step(self):
        mujoco.mj_step(self.agt_model, self.agt_data)

    def render(self, viewer):

        start_tm = time.time()

        step_start_tm = time.time()  # create point which is the first running time

        if not self.paused:
            # execute simulation if not paused
            # Every time mj_step is called, MuJoCo calculates a new state of the physical model.
            mujoco.mj_step(self.agt_model, self.agt_data)

        # Cập nhật tùy chọn hiển thị mỗi 2 giây
        # mjVIS_CONTACTPOINT : Cờ hiển thị các điểm tiếp xúc (contact points) trong mô phỏng
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.agt_data.time % 2)

        # Đồng bộ trạng thái viewer và môi trường
        # Synchronize viewer and environment state
        viewer.sync()

        # Điều chỉnh thời gian chờ để duy trì khung hình
        # Adjust the timeout to maintain frame
        # time.time() - step_start_tm : thời gian bắt đầu tính toán để mô phỏng đến hiện tại
        # self.env_model.opt.timestep : thời gian chạy mô phỏng cho mỗi bước
        # nếu < 0 => tính toán nhanh hơn mô phỏng -> chờ để mô phỏng diễn ra xong
        time_until_next_step = self.agt_model.opt.timestep - (time.time() - step_start_tm)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    # ========================= Add a new env =========================
    def add_env(self, env, env_name):
        # check the agent type
        if not isinstance(env, Environment):
            raise TypeError(f'Warning: the "agent" parameter must be an Environment object!')

        # check the agent exists or not
        if env in self.envs:
            raise ValueError(f'Warning: The agent named "{env_name}" existed!')

        # check the XML path of env
        if not os.path.exists(env.env_xml):
            raise FileNotFoundError(
                f"Warning: Having not found the XML file for the agent with the path is '{env.env_xml}!'.")

        # Thêm thông tin include vào file XML của môi trường
        self.__include_env_in_xml(env.env_xml, env_name)

        # Lưu agent vào danh sách env
        self.envs[env_name] = env
        print(f"Success: Môi trường '{env_name}' đã được thêm vào tác nhân!")

    # include an env in XML file
    def __include_env_in_xml(self, env_xml, env_name):
        # Phân tích file XML
        parser = etree.XMLParser(remove_blank_text=False, remove_comments=False)
        tree = etree.parse(self.agt_xml, parser)
        root = tree.getroot()

        # Tìm phần tử <include> có thuộc tính 'name' là env_name
        existing_include = None
        for elem in root.xpath("include"):
            if elem.get("name") == env_name:
                existing_include = elem
                break

        # Nếu phần tử <include> với agent_name đã tồn tại, xóa nó
        if existing_include is not None:
            root.remove(existing_include)

        # Tạo phần tử <include> mới với thuộc tính 'file' và 'name'
        include_element = etree.Element("include", file=env_xml.split("/")[-1])
        include_element.attrib['name'] = env_name  # Thiết lập tên cho agent
        include_element.tail = "\n"  # Thêm dấu xuống dòng

        # Thêm phần tử include_element vào vị trí đầu của root
        root.insert(0, include_element)

        # Ghi lại file XML (giữ nguyên format và comment)
        with open(self.agt_xml, "wb") as f:
            tree.write(f, pretty_print=True, encoding="utf-8", xml_declaration=True)

    # ========================= OPERATION INFORMATION ======================
    # trả về tên và index của actuator của agent
    def get_actuators_map(self):
        actuator_map = {}
        for i in range(self.agt_model.nu):  #'nu' là số actuator
            name = mujoco.mj_id2name(self.agt_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                actuator_map[name] = i

        return actuator_map

    # trả về tên và index của sensor của agent
    def get_sensors_map(self):
        sensor_map = {}
        for i in range(self.agt_model.nsensor):
            name = mujoco.mj_id2name(self.agt_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                sensor_map[name] = i

        return sensor_map

    # trả về index của actuator của agent theo tên
    def __get_actuator_name2id(self, name):
        return mujoco.mj_name2id(self.agt_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    # trả về index của sensor của agent theo tên
    def __get_sensor_name2id(self, name):
        return mujoco.mj_name2id(self.agt_model, mujoco.mjtObj.mjOBJ_SENSOR, name)

    # ========================= STATE =========================

    def set_state(self, time_step):
        # Tạo một đối tượng mới để lưu trữ trạng thái S_t
        self.S_t = AgentState(
            time_step=time_step,
            isTerminalState=False,
            joint_positions=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.joint_positions]
            ]),
            joint_velocities=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.joint_velocities]
            ]),
            left_foot_touch=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.left_foot_touch]
            ]),
            right_foot_touch=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.right_foot_touch]
            ]),
            left_foot_force=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.left_foot_force]
            ]),
            right_foot_force=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.right_foot_force]
            ]),
            left_foot_speed=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.left_foot_speed]
            ]),
            right_foot_speed=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.right_foot_speed]
            ]),
            pelvis_orientation=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_orientation]
            ]),
            pelvis_velocity=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_velocity]
            ]),
            pelvis_angular_velocity=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_angular_velocity]
            ]),
            pelvis_linear_acceleration=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_linear_acceleration]
            ]),
        )
        # Lưu giá trị actuator hiện tại
        self.atr_t[ActuatorFields.actuator_positions] = np.array(
            [self.agt_data.sensordata[i] for i in self.atr_map[ActuatorFields.actuator_positions]])
        self.atr_t[ActuatorFields.actuator_velocities] = np.array(
            [self.agt_data.sensordata[i] for i in self.atr_map[ActuatorFields.actuator_velocities]])

        # Kiểm tra xem state hiện tại có phải là state terminal không

    # ============================== MAPPING SENSOR ========================
    # Tạo mapping cho state fields và sensors
    def __get_map_name2id(self, dataClass):
        state_map_dict = {}

        # Lặp qua các trường trong StateFields
        for state_field in dataClass:
            # Lấy danh sách các cảm biến của trường
            sensor_names = state_field.value
            state_map_dict[state_field] = []  # Khởi tạo danh sách phẳng cho trường này

            for sensor_name in sensor_names:
                # Tìm ID của cảm biến từ sensors_map
                state_id = self.sensors_map.get(sensor_name, None)
                if state_id is not None:
                    # Lấy thông tin về kích thước (dim) và vị trí (adr) của cảm biến
                    state_dim = self.agt_model.sensor_dim[state_id]
                    state_offset = int(self.agt_model.sensor_adr[state_id])

                    # Tạo danh sách các chỉ số cảm biến
                    if state_dim > 1:
                        sensor_index = list(range(state_offset, state_offset + state_dim))
                    else:
                        sensor_index = [state_offset]

                    # Mở rộng danh sách phẳng
                    state_map_dict[state_field].extend(sensor_index)
                else:
                    # Cảm biến không tồn tại
                    print(f"Sensor '{sensor_name}' not found in sensors_map.")

        return state_map_dict

    # ========================= ACTION CONTROL =========================
    def extract_dTarget_ranges(self):
        dTarget_ranges = {}
        for i in range(self.agt_model.njnt):
            joint_name = mujoco.mj_id2name(self.agt_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name in JointFields.joint_positions.value:
                dTargetRanges = self.agt_model.jnt_range[i]
                dTarget_ranges[joint_name] = tuple(dTargetRanges)
                # print(joint_name, ": ", tuple(dTargetRanges))
        return dTarget_ranges

    def extract_ctrl_ranges(self):
        """
        Trích xuất ctrlrange từ mô hình đã tải của MuJoCo.

        Args:
            model (MjModel): Mô hình MuJoCo đã được tải.

        Returns:
            dict: Dictionary chứa tên actuator và ctrlrange của nó.
        """
        ctrl_ranges = {}
        for i in range(self.agt_model.nu):  # model.nu: Số lượng actuator
            name = mujoco.mj_id2name(self.agt_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            ctrlRange = self.agt_model.actuator_ctrlrange[i]
            ctrl_ranges[name] = tuple(ctrlRange)
        return ctrl_ranges

    # Tính toán torque
    def control_signal_complex(self, mu, sigma, q, qd, exploration_noise_std=1e-1):

        """
        Hàm tính toán torque điều khiển từ các giá trị mu và sigma dựa trên số actuator.

        Args:
            mu (torch.Tensor): Tensor chứa các giá trị mean (mu) đầu ra từ Actor.
            sigma (torch.Tensor): Tensor chứa các giá trị standard deviation sigma đầu ra từ Actor.
            current_actuator (int): Số lượng actuator.
            q (torch.Tensor): Vị trí hiện tại của các actuator (tensor kích thước [num_actuator]).
            qd (torch.Tensor): Vận tốc hiện tại của các actuator (tensor kích thước [num_actuator]).
            ctrl_ranges (dict): Dictionary chứa giới hạn của các actuator (vd: {"left-hip-roll": (-4.5, 4.5), ...}).

        Returns:
            torch.Tensor: Torque sau khi tính toán và clip theo `ctrl_ranges`.
            :param exploration_noise_std: Hệ số noise cho phép agent không phụ thuộc quá nhiều
            vào sigma -> tăng khám phá
        """

        # Kiểm tra điều kiện đầu vào
        assert mu.shape[-1] == self.atr_num * 3, "Số lượng mu không khớp với số actuator * 4."
        assert sigma.shape[-1] == self.atr_num * 3, "Số lượng sigma không khớp với số actuator * 4."
        assert q.shape[0] == self.atr_num, "Số actuator trong q không khớp với current_actuator."
        assert qd.shape[0] == self.atr_num, "Số actuator trong qd không khớp với current_actuator."

        # Tách mu và sigma
        pTarget_mu, dTarget_mu = mu[: self.atr_num], mu[: self.atr_num]
        pGain_mu, dGain_mu = mu[self.atr_num:self.atr_num * 2], mu[self.atr_num * 2:self.atr_num * 3]

        pTarget_sigma, dTarget_sigma = (sigma[self.atr_num],
                                        sigma[self.atr_num])

        pGain_sigma, dGain_sigma = (sigma[self.atr_num:self.atr_num * 2],
                                    sigma[self.atr_num * 2:self.atr_num * 3])

        # Lấy mẫu từ phân phối

        dist_pTarget = torch.distributions.Normal(pTarget_mu, pTarget_sigma)
        sampled_pTarget = dist_pTarget.sample()
        # dist_dTarget = torch.distributions.Normal(dTarget_mu, dTarget_sigma)
        # sampled_dTarget = dist_dTarget.sample()
        dist_pGain = torch.distributions.Normal(pGain_mu, pGain_sigma)
        sampled_pGain = dist_pGain.sample()
        dist_dGain = torch.distributions.Normal(dGain_mu, dGain_sigma )
        sampled_dGain = dist_dGain.sample()

        # Tính toán torque
        torque = []
        for i in range(self.atr_num):
            torque_i = sampled_pGain[i] * (sampled_pTarget[i] - q[i]) + \
                       sampled_dGain[i] * (0 - qd[i])
            torque.append(torque_i)

        # Clip torque về ctrl_ranges
        torque = torch.tensor(torque)
        torque = torch.clamp(torque, self.ctrl_min, self.ctrl_max)

        # Lưu lại action đã được lấy mẫu từ mu và sigma để điều khiển agent
        action = torch.cat([sampled_pTarget.squeeze(), sampled_pGain.squeeze(), sampled_dGain.squeeze()], dim=0)

        # print(f"MU_dTarget: {mu[:10]}\n"
        #       f"range_dTarget: \n{np.array([[-0.26179939, 0.39269908],
        #                                     [-0.39269908, 0.39269908],
        #                                     [-0.87266463, 1.3962634],
        #                                     [-2.86233997, -0.64577182],
        #                                     [-2.44346095, -0.52359878],
        #                                     [-0.39269908, 0.26179939],
        #                                     [-0.39269908, 0.39269908],
        #                                     [-0.87266463, 1.3962634],
        #                                     [-2.86233997, -0.64577182],
        #                                     [-2.44346095, -0.52359878]])}\n"
        #       f"dCurrent: {q}"
        #
        #       f"MU_dGain: {mu[10:20]}\n"
        #       f"MU_pGain: {mu[20:30]}\n"
        #
        #       f"SMG_dTarget: {sigma[:10]}\n"
        #       f"SMG_dGain: {sigma[10:20]}\n"
        #       f"SMG_pGain: {sigma[20:30]}\n"

        # print(f"control: {torque}\n")
              # f"range_control: \n{np.array([[-4.5, 4.5],
              #                               [-4.5, 4.5],
              #                               [-12.2, 12.2],
              #                               [-12.2, 12.2],
              #                               [-0.9, 0.9],
              #                               [-4.5, 4.5],
              #                               [-4.5, 4.5],
              #                               [-12.2, 12.2],
              #                               [-12.2, 12.2],
              #                               [-0.9, 0.9]])}\n")
        # print("===============================")

        return torque, action

    # Hàm đơn giản hơn với đầu ra của mạng là 30 tham số
    def control_signal(self, mu, q, qd):
        """
        Hàm tính toán torque điều khiển từ các giá trị mu và sigma cố định (sigma=0.1).

        Args:
            mu (torch.Tensor): Tensor chứa 30 giá trị mean (mu), bao gồm pTarget, pGain, dGain.
            q (torch.Tensor): Vị trí hiện tại của các actuator (tensor kích thước [num_actuator]).
            qd (torch.Tensor): Vận tốc hiện tại của các actuator (tensor kích thước [num_actuator]).

        Returns:
            torch.Tensor: Torque sau khi tính toán và clip theo `ctrl_ranges`.
        """
        # Kiểm tra điều kiện đầu vào
        assert mu.shape[-1] == 30, "Số lượng mu không đúng, phải là 30 (10 pTarget, 10 pGain, 10 dGain)."
        assert q.shape[0] == self.atr_num, "Số actuator trong q không khớp với current_actuator."
        assert qd.shape[0] == self.atr_num, "Số actuator trong qd không khớp với current_actuator."
        # Tách mu
        pTarget_mu = mu[:, :, :self.atr_num]  # 10 giá trị đầu tiên
        pGain_mu = mu[:, :, self.atr_num:self.atr_num * 2]  # 10 giá trị tiếp theo
        dGain_mu = mu[:, :, self.atr_num * 2:]  # 10 giá trị cuối cùng

        # Đặt sigma cố định
        sigma = 0.1

        # Lấy mẫu từ phân phối Gaussian với sigma cố định
        dist_pTarget = torch.distributions.Normal(pTarget_mu, sigma)
        sampled_pTarget = dist_pTarget.sample()

        dist_pGain = torch.distributions.Normal(pGain_mu, sigma)
        sampled_pGain = dist_pGain.sample()

        dist_dGain = torch.distributions.Normal(dGain_mu, sigma)
        sampled_dGain = dist_dGain.sample()

        # Tính toán torque
        torque = []
        for i in range(self.atr_num):
            torque_i = sampled_pGain[0, 0, i] * (sampled_pTarget[0, 0, i] - q[i]) + \
                       sampled_dGain[0, 0, i] * (0 - qd[i])  # dTarget mặc định là 0

            torque.append(torque_i)

        # Clip torque về ctrl_ranges
        torque = torch.tensor(torque)
        torque = torch.clamp(torque, self.ctrl_min, self.ctrl_max)

        return torque

    # Điều khiển agent
    def control_agent(self, control_signal):
        # print("===============BEGIN===============================")
        # print(f"Max torque: {control_signal.max()}, Min torque: {control_signal.min()}")
        # print("===============END===================================")
        for i, (torque, idx) in enumerate(zip(control_signal, self.atr_ctrl_map.values())):
            self.agt_data.ctrl[idx] = torque

    # ========================= REWARD AND CLOCK =========================

    # Tạo clock input, tính toán giá trị phạt cho mỗi pha E_C_frc, E_C_spd
    def set_clock(self, r, theta_left, theta_right, N=100, kappa=20.0, L=1):
        self.p, \
            self.ECfrc_left, \
            self.ECfrc_right, \
            self.ECspd_left, \
            self.ECspd_right = get_clock(r=r,
                                         theta_left=theta_left,
                                         theta_right=theta_right,
                                         N=N,
                                         kappa=kappa,
                                         L=L)
        self.r = r

    # ========================= GET INPUT AND OUTPUT FOR MODEL =========================

    # Lấy đầu vào cho mạng bao gồm trạng thái S_t
    def get_state_t_input_model(self, x_des, y_des, time_clock):
        S_t = torch.cat(self.S_t.as_tensor_list(), dim=0)
        xy_des = torch.tensor([x_des, y_des])
        r = torch.tensor([self.r])
        p = torch.tensor(self.p[time_clock])
        inputs = torch.cat([S_t, xy_des, r, p], dim=0)
        # Chuyển về mảng 3D mới vào LTSM được (batch_size, sequence_length, input_size)
        return inputs.unsqueeze(0).unsqueeze(0)

    def plot_EC(self):
        X_phi = np.arange(100)  # Tạo mảng từ 0 đến 99
        # Tạo hình với kích thước 12x8
        plt.figure(figsize=(12, 8))

        ''' ===================== FORCE ============================'''

        # Subplot 1: FORCE LEFT
        # 4 hàng , vị trí cột, thứ tự hàng
        plt.subplot(4, 1, 1)  # Vị trí: hàng 1, cột 1
        plt.plot(X_phi, self.ECfrc_left, label='E[C_frc(ϕ+θ)] (LEFT)')
        plt.xlabel("X (ϕ)")
        plt.ylabel("Y_spd_left")
        plt.title("FORCE - LEFT")
        plt.xlim(X_phi[0], X_phi[-1])
        plt.legend()
        plt.grid()

        # Subplot 2: FORCE RIGHT
        plt.subplot(4, 1, 2)  # Vị trí: hàng 2, cột 1
        plt.plot(X_phi, self.ECfrc_right, label='E[C_frc(ϕ+θ)] (RIGHT)')
        plt.xlabel("X (ϕ)")
        plt.ylabel("Y_spd_right")
        plt.title("FORCE - RIGHT")
        plt.xlim(X_phi[0], X_phi[-1])
        plt.legend()
        plt.grid()

        ''' ===================== SPEED ============================'''

        # Subplot 3: SPEED LEFT
        plt.subplot(4, 1, 3)  # Vị trí: hàng 3, cột 1
        plt.plot(X_phi, self.ECspd_left, label='E[C_spd(ϕ+θ)] (LEFT)')
        plt.xlabel("X (ϕ)")
        plt.ylabel("Y_spd_left")
        plt.title("SPEED - LEFT")
        plt.xlim(X_phi[0], X_phi[-1])
        plt.legend()
        plt.grid()

        # Subplot 4: SPEED RIGHT
        plt.subplot(4, 1, 4)  # Vị trí: hàng 4, cột 1
        plt.plot(X_phi, self.ECspd_right, label='E[C_spd(ϕ+θ)] (RIGHT)')
        plt.xlabel("X (ϕ)")
        plt.ylabel("Y_spd_right")
        plt.title("SPEED - RIGHT")
        plt.xlim(X_phi[0], X_phi[-1])
        plt.legend()
        plt.grid()

        # Tự động căn chỉnh khoảng cách giữa các subplot
        plt.tight_layout()

        # Hiển thị tất cả các đồ thị trên cùng một hình
        plt.show()
