# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================
import sys

import numpy as np
import mujoco
import torch

from env.agent.fall_agent import FallDetector
from env.agent.mujoco_agt import Agent
from interface.interface_main import process_action
from models.ppo_model import Actor, Critic, find_latest_model


def mujoco_viewer_process(agt: Agent, policy_freq=40, pd_control_freq=2000, use_cuda=True):
    # ------------------ SETUP TIME CONTROL --------------------
    agt.agt_model.opt.timestep = 1 / 500  # Set timestep to achieve 2000Hz
    # Định nghĩa tần số
    policy_freq = policy_freq  # Tần số chính sách (Hz) Trong 1s thu thập được 40 trạng thái
    pd_control_freq = pd_control_freq  # Tần số bộ điều khiển PD (Hz)
    steps_per_policy = pd_control_freq // policy_freq  # Số bước PD trong mỗi bước chính sách
    steps_per_policy_counter = 0  # Đếm số bước mô phỏng
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
    check_begin_state = True  # Thiết lập vị trí ban đầu
    # ------------------ SETUP CONTROL SIGNAL --------------------
    is_control = True  # Kiểm tra xem đã điều khiển agent chưa
    # Create clock for agent and control
    num_clock = 100
    theta_left = 0
    theta_right = 0.5
    r = 0.6
    agt.set_clock(r=r, N=num_clock, theta_left=theta_left, theta_right=theta_right)
    agt.x_des_vel = 0
    agt.y_des_vel = 0

    # Tạo tham số cho mô hình Actor và Critic
    path_dir = "models/param/"
    traj_input_size = 58
    traj_output_size = 50
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size,
                       pTarget_range=agt.atr_ctrl_ranges).to(device)

    with mujoco.viewer.launch_passive(agt.agt_model, agt.agt_data, show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running():
            agt.render(viewer)
            # Cứ sau num_clock timesteps thì lại đếm lại
            if timestep_clock_counter < num_clock:

                # -------- Điều khiển agent khi ở trạng thái S_t --------
                if is_control:
                    # Tải Actor và Critic mới nhất
                    path_viewer = "viewer_" + "actor_epoch_latest"
                    actor_path = find_latest_model(path_viewer, directory=path_dir)
                    if actor_path:
                        Actor_traj.load_model(actor_path)
                        # print(f"Loaded latest Actor model: {actor_path}")
                    action, mu, sigma, control = process_action(
                        agent=agt,
                        actor=Actor_traj,
                        sample_id=None,
                        tms_clk=timestep_clock_counter
                    )
                    if check_begin_state:
                        agt.set_state_begin()
                        check_begin_state = False
                    # Thiết lập mô men điều khiển cho agent

                    agt.control_agent(control)
                    is_control = False

                # -------- Tiếp tục mô phỏng quá trình điều khiển --------
                if steps_per_policy_counter < steps_per_policy:
                    steps_per_policy_counter += 1  # Tăng biến đếm lên 1 sau khi mô phỏng được 1 bước
                    continue

                steps_per_policy_counter = 0  # Trả lại trạng thái đợi mô phỏng
                is_control = True  # Trả lại trạng thái tính toán a_t
                #  ------- Thu thập trạng thái S_t+1 và tính r_t+1-------------


# Tạo kiểm tra xem điều kiện dừng cho robot có hoạt động
def mujoco_viewer_process_external_state(agt: Agent, policy_freq=40, pd_control_freq=2000, use_cuda=True):
    # ------------------ SETUP TIME CONTROL --------------------
    agt.agt_model.opt.timestep = 1 / pd_control_freq  # Set timestep to achieve 2000Hz
    agt_model = agt.agt_model
    agt_data = agt.agt_data
    # Định nghĩa tần số
    policy_freq = policy_freq  # Tần số chính sách (Hz) Trong 1s thu thập được 40 trạng thái
    pd_control_freq = pd_control_freq  # Tần số bộ điều khiển PD (Hz)
    steps_per_policy = pd_control_freq // policy_freq  # Số bước PD trong mỗi bước chính sách
    steps_per_policy_counter = 0  # Đếm số bước mô phỏng
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
    check_begin_state = True  # Thiết lập vị trí ban đầu
    # ------------------ SETUP CONTROL SIGNAL --------------------
    is_control = True  # Kiểm tra xem đã điều khiển agent chưa
    # Create clock for agent and control
    num_clock = 100
    theta_left = 0
    theta_right = 0.5
    r = 0.6
    agt.set_clock(r=r, N=num_clock, theta_left=theta_left, theta_right=theta_right)
    agt.x_des_vel = 0
    agt.y_des_vel = 0

    # Tạo tham số cho mô hình Actor và Critic
    path_dir = "models/param/"
    traj_input_size = 58
    traj_output_size = 60
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size,
                       pTarget_range=agt.dTarget_ranges).to(device)

    # -------------- KIỂM TRA VIỆC NGÃ CỦA ROBOT (final state) -----------
    fall_detector = FallDetector(n=20, min_force_threshold=5, max_tilt_threshold=90)
    # Tải Actor và Critic mới nhất
    path_viewer = "viewer_" + "actor_epoch_latest"
    actor_path = find_latest_model(path_viewer, directory=path_dir)
    with mujoco.viewer.launch_passive(agt.agt_model, agt.agt_data, show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running():
            # agt.render(viewer)
            # Cứ sau num_clock timesteps thì lại đếm lại
            if timestep_clock_counter < num_clock:
                # -------- Điều khiển agent khi ở trạng thái S_t --------
                if is_control:
                    if actor_path:
                        Actor_traj.load_model(actor_path)
                        # print(f"Loaded latest Actor model: {actor_path}")
                    action, mu, sigma, control = process_action(
                        agent=agt,
                        actor=Actor_traj,
                        tms_clk=timestep_clock_counter
                    )
                    if check_begin_state:
                        agt.set_state_begin()
                        check_begin_state = False
                    # Kiểm tra xem có bị ngã hay không
                    if fall_detector.is_fallen():
                        print("AGENT NGÃ")
                        fall_detector.reset_data()
                        # print(len(fall_detector.tilt_angle_queue))
                        agt.set_state_begin()

                    # Thiết lập mô men điều khiển cho agent
                    agt.control_agent(control)
                    # progress_console(agt)
                    agt.set_state(0)

                    # ---- Lấy dữ liệu về lực chân và độ nghiêng và kiểm tra xem có bị ngã---
                    left_fz, right_fz = agt.get_foot_forces_z()
                    tilt_angle = agt.get_tilt_angle()
                    fall_detector.update_data(left_fz=left_fz,
                                              right_fz=right_fz,
                                              tilt_angle=tilt_angle)
                    # print(len(fall_detector.tilt_angle_queue))
                    is_control = False

                # -------- Tiếp tục mô phỏng quá trình điều khiển --------
                if steps_per_policy_counter < steps_per_policy:
                    steps_per_policy_counter += 1  # Tăng biến đếm lên 1 sau khi mô phỏng được 1 bước
                    # Thiết lập mô men điều khiển cho agent
                    agt.control_agent(control)
                    mujoco.mj_step(agt_model, agt_data)
                    viewer.sync()
                    continue

                steps_per_policy_counter = 0  # Trả lại trạng thái đợi mô phỏng
                is_control = True  # Trả lại trạng thái tính toán a_t
                #  ------- Thu thập trạng thái S_t+1 và tính r_t+1-------------
            else:
                # if check_traj_id(traj_id, 0):
                #     print("TIME CLOCK: ", timestep_clock_counter)
                timestep_clock_counter = 0  # Bắt đầu 1 timestep mới của clock

# hàm thiết lập trạng thái ban đầu
def set_state_begin(agt: Agent):
    # Đặt trạng thái ban đầu trong qpos (giá trị từ key XML)
    agt.agt_data.qpos[:] = [-0.00951869, -2.10186e-06, 0.731217, 0.997481, -6.09803e-05, -0.0709308,
                            -9.63422e-05, 0.00712554, 0.0138893, 0.741668, 0.826119, -0.00862023,
                            0.0645049, -0.559726, -1.98545, -0.00211571, 2.20938, -0.00286638,
                            -1.87213, 1.85365, -1.978, -0.00687543, -0.0135461, 0.741728, 0.826182,
                            -0.00196479, -0.057323, -0.560476, -1.98537, -0.00211578, 2.20931,
                            -0.00286634, -1.87219, 1.85371, -1.97806]
    # Cập nhật trạng thái mô phỏng
    mujoco.mj_forward(agt.agt_model, agt.agt_data)
    agt.agt_data.qvel[:] = 0  # Đặt vận tốc bằng 0
    agt.agt_data.ctrl[:] = 0  # Đặt lực điều khiển bằng 0


def get_sensor(model, data):
    """
    Trích xuất dữ liệu lực từ tất cả các cảm biến trong mô hình và lưu vào dictionary theo tên cảm biến.

    Parameters:
        model: MuJoCo model (mjcModel object)
        data: MuJoCo data (mjcData object)

    Returns:
        sensors_dict: Dictionary chứa tên cảm biến và giá trị lực tương ứng.
    """
    sensors_dict = {}

    # Duyệt qua tất cả các cảm biến trong mô hình
    for i in range(model.nsensor):
        # Lấy tên cảm biến từ ID
        sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
        if sensor_name is None:
            continue  # Bỏ qua nếu không tìm thấy tên cảm biến

        # Lấy thông tin ID của cảm biến
        sensor_id = i
        sensor_dim = model.sensor_dim[sensor_id]  # Số chiều của cảm biến
        sensor_offset = model.sensor_adr[sensor_id]  # Vị trí dữ liệu trong sensordata

        # Lấy dữ liệu lực từ sensordata
        if sensor_dim > 1:
            # Nếu cảm biến có nhiều chiều (vector)
            sensors_dict[sensor_name] = data.sensordata[sensor_offset: sensor_offset + sensor_dim].copy()
        else:
            # Nếu cảm biến có một chiều (scalar)
            sensors_dict[sensor_name] = data.sensordata[sensor_offset].copy()

    return sensors_dict


def progress_console(agt: Agent):
    sensors_data = get_sensor(agt.agt_model, agt.agt_data)
    left_foot_touch = sensors_data["left-foot-touch"]
    right_foot_touch = sensors_data["right-foot-touch"]
    left_foot_force = sensors_data["left-foot-force"]
    right_foot_force = sensors_data["right-foot-force"]
    left_foot_speed = sensors_data["left-foot-speed"]
    right_foot_speed = sensors_data["right-foot-speed"]
    #  ====================== CONSOLE BEGIN ========================
    reset = "\033[0m"  # Reset màu về mặc định
    sys.stdout.write("\033[2J\033[H")  # xóa toàn bộ màn hình và đặt lai con trỏ
    sys.stdout.write(f"Touch chân trái: {left_foot_touch}\n")
    sys.stdout.write(f"Touch chân phải: {right_foot_touch}\n")
    sys.stdout.write(f"Force chân trái: {left_foot_force*bool(left_foot_touch)}\n")
    sys.stdout.write(f"Force chân phải: {right_foot_force*bool(right_foot_touch)}\n")
    sys.stdout.write(f"Speed chân trái: {left_foot_speed*(not bool(left_foot_touch))}\n")
    sys.stdout.write(f"Speed chân phải: {right_foot_speed*(not bool(right_foot_touch))}\n")
    sys.stdout.flush()
    print()  # In một dòng trống sau khi hoàn thành
    # ======================== END CONSOLE ========================


import mujoco


def mujoco_viewer_process_begin(agt: Agent):
    paused = False

    def key_callback(keycode):
        if chr(keycode) == ' ':
            nonlocal paused
            paused = not paused

    with mujoco.viewer.launch_passive(agt.agt_model, agt.agt_data,
                                      show_left_ui=True, show_right_ui=True,
                                      key_callback=key_callback) as viewer:

        set_state_begin(agt)
        i = 0
        while viewer.is_running():
            if i < 1000:
                i += 1
            else:
                i = 0
                agt.agt_data.qvel[:] = 0  # Đặt vận tốc bằng 0
                agt.agt_data.ctrl[:] = 0  # Đặt lực điều khiển bằng 0
                progress_console(agt)

            if not paused:  # Chỉ render khi không bị tạm dừng
                agt.render(viewer)

            else:
                print("Simulation paused. You can access data here.")
                # Tại đây bạn có thể lấy thông số cần thiết từ simulation

# # -------------- TÌM GIÁ TRỊ TỐI ĐA CHO VẬN TỐC GÓC CỦA CÁC ACTUATOR ----
# agt.set_state_begin()
# max_ctrlrange = []
# for i in range(agt.agt_model.nu):  # model.nu: Số lượng actuator
#     max_ctrlrange.append(agt.agt_model.actuator_ctrlrange[i][0])  # Áp dụng điều khiển tối đa
# agt.agt_data.ctrl[:] = np.array(max_ctrlrange)
#
# his_qvel = []
# for _ in range(100):
#     his_qvel.append([agt.agt_data.qvel[idx] for idx in agt.atr_ctrl_map.values()])
#     mujoco.mj_step(agt.agt_model, agt.agt_data)
#
# his_qvel = torch.tensor(np.array(his_qvel))
# max_his_qvel = torch.max(his_qvel, dim=0).values
# print("Max velocity per joint:", max_his_qvel)
