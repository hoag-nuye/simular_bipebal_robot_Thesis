import torch
import torch.distributions as dist
import random
import mujoco
import numpy as np

from env.agent.mujoco_agt import Agent
from env.agent.buffer import ReplayCache
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
    raw_sigma = traj_output_model[:, :, _range:]
    sigma = torch.log(1 + torch.exp(raw_sigma))  # Áp dụng Softplus

    # Tính toán tín hiệu điều khiển và hành động
    control, action = agent.control_signal_complex(mu, sigma,
                                                   q=agent.atr_t[ActuatorFields.actuator_positions],
                                                   qd=agent.atr_t[ActuatorFields.actuator_velocities])

    return action, mu, sigma, control


# ============= THU THẬP TRẠNG THÁI SAU KHI ACTION ĐƯỢC THỰC HIỆN ================
def collect_and_store(agent: Agent, buffer: ReplayCache, critic: Critic,
                      traj_id: int, sample_id: int, tms_clk: int,
                      action, mu, sigma):
    """
    Thu thập trạng thái mới và lưu dữ liệu vào buffer.
    """
    agent.total_samples += 1  # Tăng số lần thu thập lên 1
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
    buffer.add_sample(
        state=inputs.squeeze(),
        action=agent.a_t,
        reward=reward,
        log_prob=log_prob.cpu().detach().numpy(),
        sigma=sigma.squeeze().cpu().detach().numpy(),
        value=critic(inputs).cpu().detach().numpy().reshape(-1)[0],
        trajectory_id=traj_id
    )

    # Cập nhật hành động a_t-1
    agent.a_t_sub1 = agent.a_t


"""
    ============================================================
    -------------------------- THU THẬP TRẢI NGHIỆM ----------------------------
    ============================================================
    """


def trajectory_collection(traj_id_start,
                          num_traj,
                          num_samples_traj,
                          num_timestep_clock,
                          num_steps_per_policy,
                          agt: Agent,
                          traj_input_size, traj_output_size,
                          buffer_path_dir,
                          use_cuda,
                          buffer: ReplayCache):
    # ------------------ CÁC BIẾN ĐẾM -----------------------------
    buffer_result = None
    traj_id = traj_id_start  # Đếm số lượng trajectory
    traj_counter = 0  # Đếm số trajectory đã được thu thập trong 1 lần thu thập (32 traj) để huấn luyện
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
    samples_of_traj_counter = 0  # Đếm số lượng sample đã thu thập được trong traj hiện tại
    steps_per_policy_counter = 0  # Đếm số lượng lần mô phỏng
    # ------------------ TẠO MẠNG ACTOR VÀ CRITIC -----------------
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size).to(device)
    Critic_traj = Critic(input_size=traj_input_size).to(device)
    # Tải Actor và Critic mới nhất
    actor_path = find_latest_model("actor_epoch_", directory=buffer_path_dir)
    critic_path = find_latest_model("critic_epoch_", directory=buffer_path_dir)
    if actor_path:
        Actor_traj.load_model(actor_path)
        print(f"Loaded latest Actor model: {actor_path}")
    if critic_path:
        Critic_traj.load_model(critic_path)
        print(f"Loaded latest Critic model: {critic_path}")
    # ------------------------ SIMULAR ----------------------
    is_control = True  # Kiểm tra xem đã điều khiển agent chưa
    # Create viewer with UI options
    # Bắt đầu đo thời gian
    # ------------ RUNNING ------------------
    action, mu, sigma = None, None, None
    is_done = False  # Kiểm tra xem đã kết thúc thu thập hay chưa
    while not is_done:
        # Thu thập được 1 trajectory thì lặp lại và dừng khi đủ số lượng traj cần thu thập
        if traj_counter < num_traj and \
                ((samples_of_traj_counter < num_samples_traj) and not agt.S_t.isTerminalState):

            # Cứ sau num_clock timesteps thì lại đếm lại
            if timestep_clock_counter < num_timestep_clock:

                # -------- Điều khiển agent khi ở trạng thái S_t --------
                if is_control:
                    action, mu, sigma, control = process_action(
                        agent=agt,
                        actor=Actor_traj,
                        sample_id=samples_of_traj_counter,
                        tms_clk=timestep_clock_counter
                    )
                    # ĐIỀU KHIỂN AGENT
                    agt.control_agent(control)
                    is_control = False

                # -------- Tiếp tục mô phỏng quá trình điều khiển --------
                if steps_per_policy_counter < num_steps_per_policy:
                    steps_per_policy_counter += 1  # Tăng biến đếm lên 1 sau khi mô phỏng được 1 bước
                    # MÔ PHỎNG
                    mujoco.mj_step(agt.agt_model, agt.agt_data)
                    continue

                steps_per_policy_counter = 0  # Trả lại trạng thái đợi mô phỏng
                is_control = True  # Trả lại trạng thái tính toán a_t
                #  ------- Thu thập trạng thái S_t+1 và tính r_t+1-------------

                # Thu thập 1 sample
                collect_and_store(agent=agt,
                                  buffer=buffer,
                                  critic=Critic_traj,
                                  traj_id=traj_id,
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
            # Kêt thúc quá trình thu thập N trải nghiệm
            if traj_counter >= num_traj:
                is_done = True
                # print("__DONE__")
                buffer_result = buffer


            # Kết thúc quá trình thu thập 1 trải nghiệm
            elif not ((samples_of_traj_counter < num_samples_traj) and not agt.S_t.isTerminalState):
                samples_of_traj_counter = 0  # Bắt đầu 1 trajectory mới
                traj_counter += 1  # Bắt đầu 1 trajectory mới
                traj_id += 1  # Bắt đầu 1 trajectory mới
                # Tín hiệu điều khiển mới
                agt.x_des_vel = random.uniform(-1.5, 1.5)  # Random từ -1.5 đến 1.5 m/s
                agt.y_des_vel = random.uniform(-1.0, 1.0)  # Random từ -1.0 đến 1.0 m/s

    return buffer_result


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
