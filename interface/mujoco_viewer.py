import mujoco
import torch

from env.agent.mujoco_agt import Agent
from interface.interface_main import process_action
from models.ppo_model import Actor, Critic, find_latest_model


def mujoco_viewer_process(agt: Agent, policy_freq=40, pd_control_freq=2000, use_cuda=True):
    # ------------------ SETUP TIME CONTROL --------------------
    agt.agt_model.opt.timestep = 1/pd_control_freq  # Set timestep to achieve 2000Hz
    # Định nghĩa tần số
    policy_freq = policy_freq  # Tần số chính sách (Hz) Trong 1s thu thập được 40 trạng thái
    pd_control_freq = pd_control_freq  # Tần số bộ điều khiển PD (Hz)
    steps_per_policy = pd_control_freq // policy_freq  # Số bước PD trong mỗi bước chính sách
    steps_per_policy_counter = 0  # Đếm số bước mô phỏng
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
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
    traj_input_size = 42
    traj_output_size = 60
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size).to(device)

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
