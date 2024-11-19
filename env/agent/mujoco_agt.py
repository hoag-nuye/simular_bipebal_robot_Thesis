import time

import mujoco
import mujoco.viewer
import os

from xml.etree import ElementTree as ET
from env.mujoco_env import Environment

from env.agent.dataclass_agt import SensorFields


class Agent:
    # --------------Initial environment--------------------
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
        self.qpos_map = self.agt_data.qpos
        self.qvel_map = self.agt_data.qvel
        self.sensors_map = {}
        self.atr_map = {}
        self.sensors_data = []

        # Setup for agent
        self.__setup()

    # Save important info of MjData
    def __setup(self):
        self.sensors_map = self.get_sensors_map()
        self.atr_map = self.get_actuators_map()
        self.sensors_data = self.agt_data.sensordata
        print('Setup is done !')


    # --------------Show simulation--------------------
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

    # --------------------Add a new env----------------------
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
        tree = ET.parse(self.agt_xml)
        root = tree.getroot()

        # Tìm phần tử <include> có thuộc tính 'name' là env_name
        existing_include = None
        for elem in root.findall("include"):
            if elem.get("name") == env_name:
                existing_include = elem
                break

        # Nếu phần tử <include> với agent_name đã tồn tại, xóa nó
        if existing_include is not None:
            root.remove(existing_include)

        # Tạo phần tử <include> mới với thuộc tính 'file' và 'name'
        include_element = ET.Element("include", file=env_xml.split("/")[-1])
        include_element.attrib['name'] = env_name  # Thiết lập tên cho agent
        include_element.tail = "\n"  # Thêm dấu xuống dòng

        # Thêm phần tử include_element vào vị trí đầu của root
        root.insert(0, include_element)

        # Ghi lại file XML
        tree.write(self.agt_xml, encoding="utf-8", xml_declaration=True)

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


    # ---------------------------- STATE --------------------------
    # trả về index của actuator của agent theo tên
    def __get_actuator_name2id(self, name):
        return mujoco.mj_name2id(self.agt_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    # trả về index của sensor của agent theo tên
    def __get_sensor_name2id(self, name):
        return mujoco.mj_name2id(self.agt_model, mujoco.mjtObj.mjOBJ_SENSOR, name)

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

# ---------------------------- ACTION(CONTROL) ----------------------
# def control(self, action):
#     """
#     Điều khiển các động cơ của agent dựa trên action đầu vào.
#     Input:
#         action (dict): Dictionary chứa 10 thông số điều khiển, mỗi key là tên động cơ.
#     """
#     # Gán các giá trị điều khiển cho động cơ
#     self.env_data.ctrl[self.actuator_name_to_index("left-hip-roll")] = action["left-hip-roll"]
#     self.env_data.ctrl[self.actuator_name_to_index("left-hip-yaw")] = action["left-hip-yaw"]
#     self.env_data.ctrl[self.actuator_name_to_index("left-hip-pitch")] = action["left-hip-pitch"]
#     self.env_data.ctrl[self.actuator_name_to_index("left-knee")] = action["left-knee"]
#     self.env_data.ctrl[self.actuator_name_to_index("left-foot")] = action["left-foot"]
#     self.env_data.ctrl[self.actuator_name_to_index("right-hip-roll")] = action["right-hip-roll"]
#     self.env_data.ctrl[self.actuator_name_to_index("right-hip-yaw")] = action["right-hip-yaw"]
#     self.env_data.ctrl[self.actuator_name_to_index("right-hip-pitch")] = action["right-hip-pitch"]
#     self.env_data.ctrl[self.actuator_name_to_index("right-knee")] = action["right-knee"]
#     self.env_data.ctrl[self.actuator_name_to_index("right-foot")] = action["right-foot"]
