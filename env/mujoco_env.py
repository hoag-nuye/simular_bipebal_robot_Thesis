import time

import mujoco
import mujoco.viewer

import os

from xml.etree import ElementTree as ET
from env.mujoco_agt import Agent


class Environment:
    # --------------Initial environment--------------------
    def __init__(self, env_xml, show_left_ui=True, show_right_ui=True):
        self.env_xml = env_xml
        self.env_model = mujoco.MjModel.from_xml_path(env_xml)  # Provides the static structure of the model and is instantiated once from an XML file.
        self.env_data = mujoco.MjData(self.env_model) # stores state and is updated throughout the simulation

        # save paused action of simulation
        self.paused = False

        # Create viewer with UI options
        self.viewer = mujoco.viewer.launch_passive(
            self.env_model,
            self.env_data,
            # key_callback=self.key_callback,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui
        )

        # Create agent list
        self.agents = {}

    # --------------Show simulation--------------------
    def render(self):
        start_tm = time.time()
        # Open view during 30s
        while self.viewer.is_running() and time.time() - start_tm < 30:
            step_start_tm = time.time() # create point which is the first running time

            if not self.paused:
                # execute simulation if not paused
                # Every time mj_step is called, MuJoCo calculates a new state of the physical model.
                mujoco.mj_step(self.env_model, self.env_data)

            # Cập nhật tùy chọn hiển thị mỗi 2 giây
            # mjVIS_CONTACTPOINT : Cờ hiển thị các điểm tiếp xúc (contact points) trong mô phỏng
            with self.viewer.lock():
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.env_data.time % 2)

            # Đồng bộ trạng thái viewer và môi trường
            # Synchronize viewer and environment state
            self.viewer.sync()

            # Điều chỉnh thời gian chờ để duy trì khung hình
            # Adjust the timeout to maintain frame
            # time.time() - step_start_tm : thời gian bắt đầu tính toán để mô phỏng đến hiện tại
            # self.env_model.opt.timestep : thời gian chạy mô phỏng cho mỗi bước
            # nếu < 0 => tính toán nhanh hơn mô phỏng -> chờ để mô phỏng diễn ra xong
            time_until_next_step = self.env_model.opt.timestep - (time.time() - step_start_tm)

            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # --------------------Add a new agent----------------------
    def add_agent(self, agent, agent_name):
        # check the agent type
        if isinstance(agent, Agent):
            raise TypeError(f'Warning: the "agent" parameter must be an Agent object!')

        # check the agent exists or not
        if agent in self.agents:
            raise ValueError(f'Warning: The agent named "{agent_name}" existed!')

        # check the XML path of agent
        if os.path.exists(agent.agent_xml):
            raise FileNotFoundError(f"Warning: Having not found the XML file for the agent with the path is '{agent.agent_xml}!'.")

        # Thêm thông tin include vào file XML của môi trường
        self.__include_agent_in_xml(agent.agent_xml, agent_name)

        # Lưu agent vào danh sách agents
        self.agents[agent_name] = agent
        print(f"Success: Agent '{agent_name}' đã được thêm vào môi trường!")

        # Reload lại model để cập nhật với agent mới
        self.__reload_model()

    # include an agent in XML file
    def __include_agent_in_xml(self, agent_xml, agent_name):
        # Phân tích env xml
        tree = ET.parse(self.env_xml)
        # tìm root của agent
        root = tree.getroot(self.env_xml)
        # create element
        include_element = ET.Element("include", file=agent_xml)
        include_element.attrib['name'] = agent_name # tạo tên cho agent
        # find 'worldbody' element to add agent
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("Không tìm thấy worldbody trong XML của môi trường.")
        worldbody.append(include_element)
        # write again xml file of env
        tree.write(self.env_xml)

    def __reload_model(self):
        # Cập nhật lại model và data của môi trường sau khi thêm agent
        self.env_model = mujoco.MjModel.from_xml_path(self.env_xml)
        self.env_data = mujoco.MjData(self.env_model)
        self.viewer = mujoco.viewer.MjViewer(self.env_model, self.env_data)









