import os
import time
import mujoco
import mujoco.viewer

import torch
import numpy as np

from lxml import etree
from env.mujoco_env import Environment

from env.agent.support import get_clock
from env.agent.dataclass_agt import SensorFields, AgentState, StateFields, ActuatorFields


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

        # Save variable of reward (or different meaning is control signal)
        self.ECfrc_left = None
        self.ECfrc_right = None
        self.ECspd_left = None
        self.ECspd_right = None

        #  Save info of present state
        self.S_t: AgentState = None  # Trạng thái S_t
        self.a_t = None  # Hành động a_t
        self.V_t = None  # Ước tính giá trị S_t : V_t
        self.a_t_sub1 = []  # Hành động a_t+1
        self.S_t_add1 = None  # Trạng thái S_t+1
        self.R_t_add1 = None  # Phần thưởng R_t+1

        self.atr_t = None

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
        self.atr_map = self.get_actuators_map()
        self.state_map = self.__get_map_name2id(StateFields)
        self.atr_map = self.__get_map_name2id(ActuatorFields)
        self.atr_ctrl_map = self.get_actuators_map()
        print('Setup is done !')

    # ========================= Show simulation =========================
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

    def get_state(self, time_step):
        # Tạo một đối tượng mới để lưu trữ trạng thái S_t
        self.S_t = AgentState(
            time_step=time_step,
            joint_positions=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.joint_positions]
            ]),
            joint_velocities=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.joint_velocities]
            ]),
            left_foot_force=np.array(np.linalg.norm([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.left_foot_force]
            ])),
            right_foot_force=np.array(np.linalg.norm([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.right_foot_force]
            ])),
            pelvis_orientation=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_orientation]
            ]),
            pelvis_velocity=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_velocity]
            ]),
            pelvis_linear_acceleration=np.array([
                self.agt_data.sensordata[i] for i in self.state_map[StateFields.pelvis_linear_acceleration]
            ]),
        )
        # Lưu giá trị actuator hiện tại
        self.atr_t = np.array([self.agt_data.sensordata[i] for i in self.atr_map[ActuatorFields.actuator_positions]])

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

    def get_sensors_info(self):
        """
        Lấy thông tin trạng thái của agent từ các cảm biến của MuJoCo.
        Trả về một dictionary chứa thông số trạng thái của agent.
        """
        sensors_dict = {}

        # Lặp qua các cảm biến định nghĩa trong SensorFields
        for sensor_field in SensorFields:
            sensor_name = sensor_field.value
            sensor_id = self.sensors_map[sensor_name]

            # Lấy thông tin về kích thước (dim) và vị trí (adr) của cảm biến
            sensor_dim = self.agt_model.sensor_dim[sensor_id]
            sensor_offset = self.agt_model.sensor_adr[sensor_id]

            # Nếu cảm biến có nhiều giá trị, trích xuất tất cả giá trị
            if sensor_dim > 1:
                sensors_dict[sensor_name] = self.agt_data.sensordata[sensor_offset: sensor_offset + sensor_dim]
            else:
                sensors_dict[sensor_name] = self.agt_data.sensordata[sensor_offset]

        return sensors_dict

    # ========================= ACTION CONTROL =========================

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
    # ========================= SETUP INPUT AND OUTPUT FOR MODEL =========================

    def traj_input(self, x_des, y_des, time_clock):  # current_time: thời điểm thứ i trong clock
        S_t = torch.cat(self.S_t.as_tensor_list(), dim=0)
        xy_des = torch.tensor([x_des, y_des])
        r = torch.tensor([self.r])
        p = torch.tensor(self.p[time_clock])
        inputs = torch.cat([S_t, xy_des, r, p], dim=0)
        return inputs
