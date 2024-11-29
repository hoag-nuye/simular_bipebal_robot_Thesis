import torch
import random
import mujoco
import numpy as np

from env.mujoco_env import Environment

from env.agent.mujoco_agt import Agent
from env.agent.buffer import ReplayBuffer
from env.agent.dataclass_agt import ActuatorFields, RewardParam
from env.agent.reward import compute_reward

from models.ppo_model import Actor, Critic


# ------------- Hàm thu thập 1 sample ================
def collect_sample(agent: Agent, buffer: ReplayBuffer, actor: Actor, critic: Critic, traj_id, tms_traj, tms_clk):
    """
    Thu thập một sample từ môi trường.
        traj_id : Chỉ số của trajectory trong buffer
        tms_traj : chỉ số của sample trong 1 trajectory
        tms_clk : timestep hiện tại của clock
    """

    # Lưu dữ liệu trạng thái hiện tại vào agent
    agent.set_state(tms_traj)  # S_t đã được lấy ra và lưu lại

    # Tạo đầu vào cho Actor và Critic
    traj_input_model = agent.traj_input(agent.x_des_vel, agent.y_des_vel, tms_clk).float()
    traj_output_model = actor(traj_input_model).detach()

    # Lấy `mu` và `sigma` từ Actor để tính toán hành động
    mu = traj_output_model[:, :, :actor.output_size]
    # sigma = torch.exp(traj_output_model[:, :, 40:])

    # Điều khiển agent
    control = agent.control_signal(mu,
                                   q=agent.atr_t[ActuatorFields.actuator_positions],
                                   qd=agent.atr_t[ActuatorFields.actuator_velocities])
    agent.control_agent(control)  # Điều khiển agent
    # Chuyển tensor thành numpy array trước khi tạo np.ndarray
    agent.a_t = np.array(control.detach().cpu().numpy())  # Hành động a_t tại trạng thái S_t là tín hiệu điều khiển agent
    agent.S_t_add1 = agent.get_state()  # Lấy ngay trạng thái sau khi điều khiển

    # Tính phần thưởng (reward)
    reward = compute_reward(
        RewardParam(S_t=agent.S_t,
                    action_t=agent.a_t,
                    action_t_sub1=agent.a_t_sub1,
                    r_swing=agent.r,
                    ECfrc_left=agent.ECfrc_left,
                    ECfrc_right=agent.ECfrc_right,
                    ECspd_left=agent.ECspd_left,
                    ECspd_right=agent.ECspd_right,
                    x_des=agent.x_des_vel,
                    y_des=agent.y_des_vel,
                    quat_des=agent.S_t.pelvis_orientation)
    )

    # Lưu vào buffer
    buffer.add_sample(
        state=agent.S_t,
        action=agent.a_t,
        reward=reward,
        next_state=agent.S_t_add1,
        log_prob=mu.squeeze(),  # Lưu log_prob để PPO sử dụng
        # Tính giá trị V_t và detach việc tính toán gradient cho mạng
        # sử dụng .numpy().reshape(-1) hoặc .detach().view(-1) cho tensor
        value=critic(traj_input_model).detach().view(-1).float().item(),  # vì là mạng lstm nên chuyển từ 3D -> 1D -> float
        timestep=tms_traj,
        trajectory_id=traj_id
    )
    # Cập nhật hành động a_t-1
    agent.a_t_sub1 = agent.a_t


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
    num_traj = 32  # Số trajectory cần thu thập
    num_samples_traj = 300  # Số samples tối đa trong 1 trajectory

    # ----------------- REPLAY BUFFER --------------------
    dir_file = 'env/agent/'
    buffer = ReplayBuffer(trajectory_size=num_traj*num_samples_traj,
                          max_size=50000,
                          gamma=0.99, lam=0.95, alpha=0.6,
                          file_path=f"{dir_file}replay_buffer")

    # Setup real-time control
    steps_counter_sim = 0  # Đếm số bước mô phỏng
    steps_per_collection = int(1 / (100 * agt.agt_model.opt.timestep))  # Số bước để thu thập mẫu mỗi giây (250 Hz)

    traj_total_counter = 0  # Đếm số trajectory đã thu thập được trong suốt quá trình huấn luyện 1 policy
    traj_counter = 0  # Đếm số trajectory đã được thu thập trong 1 lần thu thập (32 traj) để huấn luyện
    timestep_clock_counter = 0  # Đếm số timestep đã đi qua trong clock (< num_clock)
    samples_of_traj_counter = 0  # Đếm số lượng đã thu thập được trong traj hiện tại

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
    traj_input_size = 42
    traj_output_size = 30
    Actor_traj = Actor(input_size=traj_input_size, output_size=traj_output_size).to(device)
    Critic_traj = Critic(input_size=traj_input_size).to(device)

    # ------------------------ SIMULAR ----------------------
    called_once = True
    # Create viewer with UI options
    with mujoco.viewer.launch_passive(agt.agt_model, agt.agt_data, show_left_ui=True, show_right_ui=True) as viewer:

        # ------------ RUNNING ------------------
        while viewer.is_running():
            agt.render(viewer)

            # Bắt đầu thu thập mẫu sau 1 khoảng thời gian nhất định
            if steps_counter_sim % steps_per_collection == 0:

                # Thu thập được 1 trajectory thì lặp lại và dừng khi đủ số lượng traj cần thu thập
                if traj_counter < num_traj and \
                        ((samples_of_traj_counter < num_samples_traj) and not agt.S_t.isTerminalState):
                    print(f"Trajectory: {traj_counter} "
                          f"- Time step: {samples_of_traj_counter} "
                          f"- Clock time: {timestep_clock_counter}")

                    # Cứ sau num_clock timesteps thì lại đếm lại
                    if timestep_clock_counter < num_clock:

                        # Thu thập 1 sample
                        collect_sample(agent=agt,
                                       buffer=buffer,
                                       actor=Actor_traj,
                                       critic=Critic_traj,
                                       traj_id=traj_total_counter,
                                       tms_traj=traj_counter,
                                       tms_clk=timestep_clock_counter)

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
                            print("Rewards: ", data_batch["rewards"])

                    elif not ((samples_of_traj_counter < num_samples_traj) and not agt.S_t.isTerminalState):
                        print("DONE A TRAJECTORY!")
                        samples_of_traj_counter = 0  # Bắt đầu 1 trajectory mới
                        traj_counter += 1  # Bắt đầu 1 trajectory mới
                        # Tín hiệu điều khiển mới
                        agt.x_des_vel = random.uniform(-1.5, 1.5)  # Random từ -1.5 đến 1.5 m/s
                        agt.y_des_vel = random.uniform(-1.0, 1.0)  # Random từ -1.0 đến 1.0 m/s

            # Đếm số bước mô phỏng
            steps_counter_sim += 1


# ================== RUN MAIN =======================
if __name__ == "__main__":
    # Bật/tắt CUDA bằng cách thay đổi giá trị `use_cuda`
    main(use_cuda=True)
