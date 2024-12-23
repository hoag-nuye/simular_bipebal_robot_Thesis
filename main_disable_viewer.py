# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import time
import torch
import torch.distributions as dist
import random
import mujoco
import numpy as np

from env.mujoco_env import Environment

from env.agent.mujoco_agt import Agent
from env.agent.buffer import ReplayBuffer
from env.agent.dataclass_agt import ActuatorFields, RewardParam
from env.agent.reward import compute_reward

from models.ppo_model import Actor, Critic, PPOClip_Training, find_latest_model

"""
============================================================
------------------ COLLECTION SUPPORT ---------------------
============================================================
"""


# ------------- Hàm tính toán pi cho hành đông a_t tại s_t --------------
def compute_log_pi(actions, mu, sigma):
    """
    Tính log-probability của các hành động dựa trên phân phối chuẩn.

    Args:
        actions: Tensor các hành động a (batch_size x action_dim)
        mu: Tensor giá trị trung bình mu
        sigma: Tensor độ lệch chuẩn sigma

    Returns:
        log_pi: Tensor log-probability
    """
    normal_dist = dist.Normal(mu, sigma)  # Tạo phân phối chuẩn với mu và sigma
    log_prob = normal_dist.log_prob(actions)  # Tính log-probability
    return log_prob.sum(dim=-1)  # Tổng log-probability trên tất cả các hành động


# ============= LẤY STATE HIỆN TẠI VÀ TÍNH TOÁN ACTION ================
def process_action(agent: Agent, actor: Actor, sample_id: int, tms_clk: int):
    """
    Xử lý trạng thái hiện tại và tính toán hành động.
    """
    if sample_id == 0:
        # Lưu dữ liệu trạng thái hiện tại vào agent
        agent.set_state(sample_id)  # S_t đã được lấy ra và lưu lại

    # Tạo đầu vào cho Actor
    traj_input_model = agent.traj_input(agent.x_des_vel, agent.y_des_vel, tms_clk).float()
    traj_output_model = actor(traj_input_model).detach()

    # Lấy `mu` và `sigma` từ Actor để tính toán hành động
    _range = int(actor.output_size / 2)
    mu = traj_output_model[:, :, :_range]
    sigma = traj_output_model[:, :, _range:]

    # Tính toán tín hiệu điều khiển và hành động
    control, action = agent.control_signal_complex(mu, sigma,
                                                   q=agent.atr_t[ActuatorFields.actuator_positions],
                                                   qd=agent.atr_t[ActuatorFields.actuator_velocities])

    return action, mu, sigma, control


# ============= THU THẬP TRẠNG THÁI SAU KHI ACTION ĐƯỢC THỰC HIỆN ================
def collect_and_store(agent: Agent, buffer: ReplayBuffer, critic: Critic,
                      traj_id: int, sample_id: int, tms_clk: int,
                      action, mu, sigma):
    """
    Thu thập trạng thái mới và lưu dữ liệu vào buffer.
    """
    # Lấy trạng thái mới từ môi trường
    agent.set_state(sample_id)

    # Chuyển tensor hành động thành numpy array
    agent.a_t = np.array(action.detach().cpu().numpy())

    # Tính phần thưởng
    reward = compute_reward(
        RewardParam(S_t=agent.S_t,
                    action_t=agent.a_t,
                    action_t_sub1=agent.a_t_sub1,
                    r_swing=agent.r,
                    tms_clk=tms_clk,
                    ECfrc_left=agent.ECfrc_left,
                    ECfrc_right=agent.ECfrc_right,
                    ECspd_left=agent.ECspd_left,
                    ECspd_right=agent.ECspd_right,
                    x_des=agent.x_des_vel,
                    y_des=agent.y_des_vel,
                    quat_des=agent.S_t.pelvis_orientation)
    )

    # Tính log_prob
    log_prob = compute_log_pi(action, mu.squeeze(), sigma.squeeze())

    # Lưu dữ liệu vào buffer
    inputs = agent.get_state_t_input_model(agent.x_des_vel, agent.y_des_vel, tms_clk).float()
    buffer.add_sample(state=inputs.squeeze(), action=agent.a_t, reward=reward, log_prob=log_prob.cpu().detach().numpy(),
                      value=critic(inputs).cpu().detach().numpy().reshape(-1)[0], returns=0, advantages=0, td_errors=0,
                      trajectory_id=traj_id)

    # Cập nhật hành động a_t-1
    agent.a_t_sub1 = agent.a_t


"""
============================================================
------------------------ TRAIN SUPPORT ---------------------
============================================================
"""

"""
"states"
"actions": 
"log_probs":
"sigma": 
"returns":
"advantages"
"""


def train(training_id, data_batch, epochs, learning_rate, output_size, path_dir, use_cuda):
    # Kiểm tra xem có CUDA hay không
    # https://pytorch.org/get-started/locally/
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_size = data_batch["states"].shape[2]
    for k, v in data_batch.items():
        # CHUYỂN TENSOR LÊN ĐÚNG THIẾT BỊ VỚI MODEL
        if isinstance(v, torch.Tensor):
            data_batch[k] = v.to(device).float()
        # print(f"Tensor{k} đang ở trên: {v.device}")

    actor = Actor(input_size=input_size, output_size=output_size).to(device)
    critic = Critic(input_size=input_size).to(device)
    ppo_clip_training = PPOClip_Training(training_id=training_id,
                                         actor_model=actor,
                                         critic_model=critic,
                                         states=data_batch["states"],
                                         actions=data_batch["actions"],
                                         log_probs=data_batch["log_probs"],
                                         sigma=data_batch["sigma"],
                                         returns=data_batch["returns"],
                                         advantages=data_batch["advantages"],
                                         epsilon=0.2,
                                         entropy_weight=0.01,
                                         num_epochs=epochs,
                                         learning_rate=learning_rate,
                                         path_dir=path_dir)

    ppo_clip_training.train()


"""
============================================================
-------------------------- MAIN ----------------------------
============================================================
"""


def main(use_cuda=False):
    # -------------- initial environment --------------------
    agt_xml_path = 'structures/agility_cassie/cassie.xml'
    agt = Agent(agt_xml_path)

    # -------------- Add a new agent ---------------------
    env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
    agt.add_env(env, 'cassie_env')
    agt.set_state(0)  # Khởi tạo giá trị ban đầu

    # ======================== THU THẬP 32 TRAJECTORY =======================
    # Tạo tham số cho lần thua thập
    num_traj = 1  # Số trajectory cần thu thập
    num_samples_traj = 300  # Số samples tối đa trong 1 trajectory

    # ----------------- REPLAY BUFFER --------------------
    dir_file = 'env/agent/'
    buffer = ReplayBuffer(trajectory_size=num_traj * num_samples_traj,
                          max_size=50000,
                          gamma=0.99, lam=0.95, alpha=0.6,
                          file_path=f"{dir_file}replay_buffer")

    # ------------------ SETUP TIME CONTROL --------------------
    agt.agt_model.opt.timestep = 0.0005  # Set timestep to achieve 2000Hz
    # Định nghĩa tần số
    policy_freq = 40  # Tần số chính sách (Hz) Trong 1s thu thập được 40 trạng thái
    pd_control_freq = 2000  # Tần số bộ điều khiển PD (Hz)
    steps_per_policy = pd_control_freq // policy_freq  # Số bước PD trong mỗi bước chính sách
    steps_per_policy_counter = 0  # Đếm số bước mô phỏng

    # ------------------ CÁC BIẾN ĐẾM -----------------------------
    train_counter = 0
    traj_total_counter = 0  # Đếm số trajectory đã thu thập được trong suốt quá trình huấn luyện 1 policy
    traj_counter = 0  # Đếm số trajectory đã được thu thập trong 1 lần thu thập (32 traj) để huấn luyện
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
    samples_of_traj_counter = 0  # Đếm số lượng sample đã thu thập được trong traj hiện tại

    is_control = True  # Kiểm tra xem đã điều khiển agent chưa

    # Create clock for agent and control
    num_clock = 100
    theta_left = 0
    theta_right = 0.5
    r = 0.6
    agt.set_clock(r=r, N=num_clock, theta_left=theta_left, theta_right=theta_right)
    agt.x_des_vel = random.uniform(-1.5, 1.5)  # Random từ -1.5 đến 1.5 m/s
    agt.y_des_vel = random.uniform(-1.0, 1.0)  # Random từ -1.0 đến 1.0 m/s

    # ----------------------- MODEL --------------------
    # Kiểm tra xem có CUDA hay không
    # https://pytorch.org/get-started/locally/
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tạo tham số cho mô hình Actor và Critic
    path_dir = "models/param/"
    traj_input_size = 42
    traj_output_size = 60
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size).to(device)
    Critic_traj = Critic(input_size=traj_input_size).to(device)
    # Tải Actor và Critic mới nhất
    actor_path = find_latest_model("actor_epoch_", directory=path_dir)
    critic_path = find_latest_model("critic_epoch_", directory=path_dir)
    if actor_path:
        Actor_traj.load_model(actor_path)
        print(f"Loaded latest Actor model: {actor_path}")
    if critic_path:
        Critic_traj.load_model(critic_path)
        print(f"Loaded latest Critic model: {critic_path}")

    # ------------------------ SIMULAR ----------------------
    called_once = True
    # Create viewer with UI options
    # Bắt đầu đo thời gian
    start_time = time.time()

    # ------------ RUNNING ------------------
    while True:
        # Thu thập được 1 trajectory thì lặp lại và dừng khi đủ số lượng traj cần thu thập
        if traj_counter < num_traj and \
                ((samples_of_traj_counter < num_samples_traj) and not agt.S_t.isTerminalState):

            # Cứ sau num_clock timesteps thì lại đếm lại
            if timestep_clock_counter < num_clock:

                # -------- Điều khiển agent khi ở trạng thái S_t --------
                if is_control:
                    action, mu, sigma, control = process_action(
                        agent=agt,
                        actor=Actor_traj,
                        sample_id=samples_of_traj_counter,
                        tms_clk=timestep_clock_counter
                    )
                    # Thiết lập mô men điều khiển cho agent
                    agt.control_agent(control)
                    is_control = False

                # -------- Tiếp tục mô phỏng quá trình điều khiển --------
                if steps_per_policy_counter < steps_per_policy:
                    steps_per_policy_counter += 1  # Tăng biến đếm lên 1 sau khi mô phỏng được 1 bước
                    mujoco.mj_step(agt.agt_model, agt.agt_data)
                    continue

                steps_per_policy_counter = 0  # Trả lại trạng thái đợi mô phỏng
                is_control = True  # Trả lại trạng thái tính toán a_t
                #  ------- Thu thập trạng thái S_t+1 và tính r_t+1-------------
                print(f"Trajectory: {traj_counter} "
                      f"- Time step: {samples_of_traj_counter} "
                      f"- Clock time: {timestep_clock_counter}")
                # Thu thập 1 sample
                collect_and_store(agent=agt,
                                  buffer=buffer,
                                  critic=Critic_traj,
                                  traj_id=traj_total_counter,
                                  sample_id=samples_of_traj_counter,
                                  tms_clk=timestep_clock_counter,
                                  action=action,
                                  mu=mu,
                                  sigma=sigma)

                timestep_clock_counter += 1
                samples_of_traj_counter += 1  # Đếm số lượng sample nhưng không reset khi hết 1 clock
            else:
                timestep_clock_counter = 0  # Bắt đầu 1 timestep mới của clock

        else:
            if traj_counter >= num_traj and called_once:  # Nếu kết thúc số lượng traj cần thu thập
                # traj_counter = 0 # Bắt đầu lần thu thập mới
                # lưu dữ liệu cũ
                print("DONE")
                if called_once:
                    buffer.compute_returns_and_advantages()
                    buffer.update_td_errors()
                    buffer.save_to_pkl()
                    buffer.reset()
                    called_once = False

                    data_batch = buffer.sample_batch(batch_size=num_traj)
                    for k, v in data_batch.items():
                        print(f"Size of {k} is: {len(v)}, shape:{v.shape} ({type(v)})")

                    """
                    ============================================================
                    QUÁ TRÌNH TRAIN DIỄN RA SAU 1 LẦN THU THẬP ĐỦ SỐ LƯỢNG BATCH
                    ============================================================
                    """

                    # train(training_id=train_counter,
                    #       data_batch=data_batch,
                    #       epochs=4,
                    #       learning_rate=0.0001,
                    #       output_size=traj_output_size,
                    #       path_dir="models/param",
                    #       use_cuda=use_cuda)
                    # train_counter += 1

                    # Kết thúc đo thời gian
                    end_time = time.time()
                    # Tính toán và in ra thời gian chạy
                    execution_time = end_time - start_time
                    print(f"Thời gian thực hiện COLLECTION & TRAIN: {execution_time} s")

            elif not ((samples_of_traj_counter < num_samples_traj) and not agt.S_t.isTerminalState):
                print("DONE A TRAJECTORY!")
                samples_of_traj_counter = 0  # Bắt đầu 1 trajectory mới
                traj_counter += 1  # Bắt đầu 1 trajectory mới
                traj_total_counter += 1  # Bắt đầu 1 trajectory mới
                # Tín hiệu điều khiển mới
                agt.x_des_vel = random.uniform(-1.5, 1.5)  # Random từ -1.5 đến 1.5 m/s
                agt.y_des_vel = random.uniform(-1.0, 1.0)  # Random từ -1.0 đến 1.0 m/s


# ================== RUN MAIN =======================
if __name__ == "__main__":
    # Bật/tắt CUDA bằng cách thay đổi giá trị `use_cuda`
    main(use_cuda=True)
