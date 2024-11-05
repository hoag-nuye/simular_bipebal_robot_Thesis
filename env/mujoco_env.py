import time

import mujoco
import mujoco.viewer

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


