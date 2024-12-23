# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================
import time

import torch
import torch.distributions as dist
import mujoco
import numpy as np

from env.agent.fall_agent import FallDetector
from env.agent.mujoco_agt import Agent
from env.agent.buffer import ReplayCache
from env.agent.dataclass_agt import ActuatorFields, RewardParam
from env.agent.reward import compute_reward

from models.ppo_model import Actor, Critic, PPOClip_Training, find_latest_model

# Hiển thị reward theo từng lan huấn luyện
# from env.agent.reward import RewardsHistory
# rewards_history = RewardsHistory()
import global_value as glv
from interface.progress_console import collect_progress_console

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
def process_action(agent: Agent, actor: Actor, tms_clk: int):
    """
    Xử lý trạng thái hiện tại và tính toán hành động.
    """
    # Tạo đầu vào cho Actor
    traj_input_model = agent.get_state_t_input_model(agent.x_des_vel, agent.y_des_vel, tms_clk).float()
    # Lấy `mu` và `sigma` từ Actor để tính toán hành động
    mu, sigma = actor(traj_input_model)
    mu = mu.detach()
    sigma = sigma.detach()

    # Tính toán tín hiệu điều khiển và hành động
    torque, action = agent.control_signal_complex(mu, sigma,
                                                   q=agent.atr_t[ActuatorFields.actuator_positions],
                                                   qd=agent.atr_t[ActuatorFields.actuator_velocities])
    return action, mu, sigma, torque


# ============= THU THẬP TRẠNG THÁI SAU KHI ACTION ĐƯỢC THỰC HIỆN ================
def collect_and_store(agent: Agent, buffer: ReplayCache, critic: Critic,
                      traj_id: int, tms_clk: int,
                      action, torque_t, mu, sigma, fall_reward):
    """
    Thu thập trạng thái mới và lưu dữ liệu vào buffer.
    """
    agent.total_samples += 1  # Tăng số lần thu thập lên 1

    # Chuyển tensor hành động thành numpy array
    agent.a_t = np.array(action.detach().cpu().numpy())
    agent.torque_t = np.array(torque_t.detach().cpu().numpy())
    # Tính phần thưởng
    reward = compute_reward(
        RewardParam(S_t=agent.S_t,
                    torque_t=torque_t,
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
                    quat_des=agent.quat_des,
                    fall_reward=fall_reward)
    )
    # print(reward)
    # print(tms_clk)
    # rewards_history.add_reward(reward, r_bipedal, r_cmd, r_smooth, r_std_cost, tms_clk)

    # Tính log_prob
    log_prob = compute_log_pi(action, mu.squeeze(), sigma.squeeze())

    # Lưu dữ liệu vào buffer
    inputs = agent.get_state_t_input_model(agent.x_des_vel, agent.y_des_vel, tms_clk).float()
    buffer.add_sample(
        state=inputs.squeeze(),
        action=agent.a_t,
        reward=reward,
        log_prob=log_prob.cpu().detach().numpy(),
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


def check_traj_id(real_id, check_id):
    return True if real_id == check_id else False


def trajectory_collection(max_sample_collect,
                          num_samples_traj,
                          num_timestep_clock,
                          num_steps_per_policy,
                          agt: Agent,
                          traj_input_size, traj_output_size,
                          param_path_dir,
                          device,
                          buffer: ReplayCache):
    # ------------------ CÁC BIẾN ĐẾM -----------------------------
    # Sử dụng lock và shared_var trong tiến trình

    num_sample = 0  # Đếm số lượng sample đã thu thập được
    traj_counter = 0  # Đếm số trajectory đã được thu thập trong 1 lần thu thập (32 traj) để huấn luyện
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
    samples_of_traj_counter = 0  # Đếm số lượng sample đã thu thập được trong traj hiện tại
    steps_per_policy_counter = 0  # Đếm số lượng lần mô phỏng
    # ------------------ TẠO MẠNG ACTOR VÀ CRITIC -----------------
    device = device
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size,
                       pTarget_range=agt.dTarget_ranges).to(device)
    Critic_traj = Critic(input_size=traj_input_size).to(device)
    # Tải Actor và Critic mới nhất
    # actor_path = find_latest_model("actor_epoch_", directory=param_path_dir)
    # critic_path = find_latest_model("critic_epoch_", directory=param_path_dir)
    actor_path = find_latest_model("actor_epoch_latest", directory=param_path_dir)
    critic_path = find_latest_model("critic_epoch_latest", directory=param_path_dir)
    if actor_path:
        Actor_traj.load_model(actor_path)
        # print(f"Loaded latest Actor model: {actor_path}")
    if critic_path:
        Critic_traj.load_model(critic_path)
        # print(f"Loaded latest Critic model: {critic_path}")

    # -------------- KIỂM TRA VIỆC NGÃ CỦA ROBOT (final state) -----------
    check_begin_state = True  # Thiết lập vị trí ban đầu
    check_terminal_state = False  # Kiểm tra trạng thái cuối
    fall_detector = FallDetector(n=30, min_force_threshold=10, max_tilt_threshold=85)

    # ------------------------ SIMULAR ----------------------
    is_control = True  # Kiểm tra xem đã điều khiển agent chưa
    # Create viewer with UI options
    # Bắt đầu đo thời gian
    # ------------ RUNNING ------------------
    start_time = time.time()
    action, mu, sigma, control = None, None, None, None
    is_done = False  # Kiểm tra xem đã kết thúc thu thập hay chưa
    while not is_done:

        # print(traj_id)
        # Thu thập được 1 trajectory thì lặp lại và dừng khi đủ số lượng traj cần thu thập
        if (samples_of_traj_counter < num_samples_traj) and \
                not check_terminal_state:

            # Cứ sau num_clock timesteps thì lại đếm lại
            if timestep_clock_counter < num_timestep_clock:

                # -------- Điều khiển agent khi ở trạng thái S_t --------
                if is_control:
                    # Thiết lập trạng thái ban đầu của agent
                    if check_begin_state:
                        agt.set_state_begin()
                        check_begin_state = False

                    agt.set_state(0)

                    # ---- Lấy dữ liệu về lực chân và độ nghiêng và kiểm tra xem có bị ngã---
                    # Kiểm tra xem có bị ngã hay không

                    left_fz, right_fz = agt.get_foot_forces_z()
                    tilt_angle = agt.get_tilt_angle()
                    fall_detector.update_data(left_fz=left_fz,
                                              right_fz=right_fz,
                                              tilt_angle=tilt_angle)
                    if fall_detector.is_fallen():
                        fall_reward = -10
                        # print(f"Number trajectory:{traj_counter} -- sample: {samples_of_traj_counter}")
                        fall_detector.reset_data()
                        check_terminal_state = True
                    else:
                        # print(f"Number trajectory:{traj_counter} -- sample: {samples_of_traj_counter}")
                        fall_reward = 0

                    #  ======== ĐIỀU KHIỂN AGENT =============
                    torque, mu, sigma, control = process_action(
                        agent=agt,
                        actor=Actor_traj,
                        tms_clk=timestep_clock_counter
                    )
                    # print(f"MU_dTarget: {mu[:10]}\n"
                    #       f"range_pTarget: \n{np.array(list(agt.dTarget_ranges.values()))}\n"
                    #       f"MU_dGain: {mu[10:20]}\n"
                    #       f"MU_pGain: {mu[20:30]}\n"
                    #
                    #       f"SMG_dTarget: {sigma[:10]}\n"
                    #       f"SMG_dGain: {sigma[10:20]}\n"
                    #       f"SMG_pGain: {sigma[20:30]}\n"
                    #
                    #       f"control: {torque}\n"
                    #       f"range_control: \n{np.array(list(agt.atr_ctrl_ranges.values()))}\n")
                    # print("===============================")
                    #  ======== Thu thập trạng thái S_t+1 và tính r_t+1 =============
                    # Thu thập 1 sample
                    collect_and_store(agent=agt,
                                      buffer=buffer,
                                      critic=Critic_traj,
                                      traj_id=traj_counter,
                                      tms_clk=timestep_clock_counter,
                                      action=torque,
                                      fall_reward=fall_reward,
                                      torque_t=control,
                                      mu=mu,
                                      sigma=sigma)

                    # --------------- THU THẬP ĐỦ 50000 SAMPLE THÌ DỪNG ----------
                    with glv.g_lock:
                        glv.g_shared_var.value += 1
                        num_sample = glv.g_shared_var.value
                        if glv.g_shared_var.value == max_sample_collect:
                            glv.g_shared_var.value -= 1
                            is_done = True
                            continue

                    # Hiển thị tiến độ thu thập
                    collect_progress_console(total_steps=max_sample_collect,
                                             current_steps=num_sample+1,
                                             begin_time=start_time)

                    timestep_clock_counter += 1
                    samples_of_traj_counter += 1  # Đếm số lượng sample nhưng không reset khi hết 1 clock
                    is_control = False

                # -------- Tiếp tục mô phỏng quá trình điều khiển --------
                if steps_per_policy_counter < num_steps_per_policy:
                    steps_per_policy_counter += 1  # Tăng biến đếm lên 1 sau khi mô phỏng được 1 bước
                    # MÔ PHỎNG
                    # ĐIỀU KHIỂN AGENT
                    agt.control_agent(control)
                    mujoco.mj_step(agt.agt_model, agt.agt_data)
                    continue
                # if check_traj_id(traj_id, 0):
                #     print("TẦN SỐ :", steps_per_policy_counter)
                steps_per_policy_counter = 0  # Trả lại trạng thái đợi mô phỏng
                is_control = True  # Trả lại trạng thái tính toán a_t

            else:
                # if check_traj_id(traj_id, 0):
                #     print("TIME CLOCK: ", timestep_clock_counter)
                timestep_clock_counter = 0  # Bắt đầu 1 timestep mới của clock

        else:
            # if check_traj_id(traj_id, 0):
            #     print("SAMPLE: ", samples_of_traj_counter)
            check_terminal_state = False  # Bắt đầu 1 trajectory mới thì ko có trang thái cuối
            check_begin_state = True  # Bắt đầu 1 trajectory mới
            samples_of_traj_counter = 0  # Bắt đầu 1 trajectory mới
            timestep_clock_counter = 0  # Bắt đầu 1 trajectory mới
            steps_per_policy_counter = 0  # Bắt đầu 1 trajectory mới
            traj_counter += 1  # Bắt đầu 1 trajectory mới


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


def train(agt: Agent, iters_passed, data_batch, is_save, epochs,
          actor_learning_rate, critic_learning_rate, entropy_weight, epsilon,
          output_size, path_dir, device):
    # Kiểm tra xem có CUDA hay không
    # https://pytorch.org/get-started/locally/
    device = device
    # print(f"Using device: {device}")
    input_size = data_batch["states"].shape[2]
    for k, v in data_batch.items():
        # CHUYỂN TENSOR LÊN ĐÚNG THIẾT BỊ VỚI MODEL
        if isinstance(v, torch.Tensor):
            data_batch[k] = v.to(device).float()

    data_batch["lengths_traj"] = data_batch["lengths_traj"].flatten()  # Đảm bảo tensor là 1D
    data_batch["lengths_traj"] = data_batch["lengths_traj"].cpu()  # Chuyển về CPU
    data_batch["lengths_traj"] = data_batch["lengths_traj"].to(torch.int64)  # Đảm bảo kiểu int64

    actor = Actor(input_size=input_size, output_size=output_size,
                  lengths_traj=data_batch["lengths_traj"],
                  pTarget_range=agt.atr_ctrl_ranges).to(device)

    critic = Critic(input_size=input_size, lengths_traj=data_batch["lengths_traj"]).to(device)
    ppo_clip_training = PPOClip_Training(iters_passed=iters_passed,
                                         actor_model=actor,
                                         critic_model=critic,
                                         states=data_batch["states"],
                                         actions=data_batch["actions"],
                                         log_probs=data_batch["log_probs"],
                                         rewards=data_batch["rewards"],
                                         returns=data_batch["returns"],
                                         advantages=data_batch["advantages"],
                                         epsilon=epsilon,
                                         entropy_weight=entropy_weight,
                                         is_save=is_save,
                                         num_epochs=epochs,
                                         actor_learning_rate=actor_learning_rate,
                                         critic_learning_rate=critic_learning_rate,
                                         path_dir=path_dir)

    actor_loss, critic_loss, mean_reward = ppo_clip_training.train()
    return actor_loss, critic_loss, mean_reward
