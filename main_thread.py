import torch
import random
import mujoco
import numpy as np
from multiprocessing import Process, Queue, Value, Lock

from env.mujoco_env import Environment
from env.agent.mujoco_agt import Agent
from env.agent.buffer import ReplayBuffer
from env.agent.dataclass_agt import ActuatorFields, RewardParam
from env.agent.reward import compute_reward
from models.ppo_model import Actor, Critic


# ======================== HÀM CHÍNH =============================
# ----------- Hàm thu thập sample (giữ nguyên từ code gốc) -------------
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
    agent.a_t = np.array(
        control.detach().cpu().numpy())  # Hành động a_t tại trạng thái S_t là tín hiệu điều khiển agent
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
        value=critic(traj_input_model).detach().view(-1).float().item(),
        # vì là mạng lstm nên chuyển từ 3D -> 1D -> float
        timestep=tms_traj,
        trajectory_id=traj_id
    )
    # Cập nhật hành động a_t-1
    agent.a_t_sub1 = agent.a_t


# ----------- Hàm huấn luyện Actor-Critic (giả định) -------------
def train_actor_critic(buffer: ReplayBuffer, actor: Actor, critic: Critic):
    """
    Huấn luyện Actor và Critic dựa trên dữ liệu từ Replay Buffer.
    Trả về tham số đã cập nhật cho Actor.
    """
    # Logic huấn luyện mô hình
    return actor.state_dict()  # Trả về tham số mới của Actor


# ============================= THREAD ===============================
# ----------- Luồng 1: Điều khiển agent -------------
def autonomous_thread(agent: Agent, queue: Queue, stop_flag: Value, lock: Lock):
    """
    Luồng 1: Điều khiển agent tự hành.
    - Nhận tham số từ luồng 2 thông qua queue.
    - Sử dụng tham số hiện tại để chạy mô phỏng.
    """
    with mujoco.viewer.launch_passive(agent.agt_model, agent.agt_data, show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running():
            agent.render(viewer)

            with lock:
                if stop_flag.value:  # Kiểm tra tín hiệu dừng từ luồng 2
                    continue  # Tạm dừng khi stop_flag bật

            # Nhận tham số từ luồng 2 (nếu có)
            if not queue.empty():
                new_params = queue.get()
                agent.update_parameters(new_params)

            # Điều khiển agent dựa trên tham số hiện tại
            control = agent.control_agent(agent.current_parameters)
            agent.control_agent(control)  # Điều khiển agent


# ----------- Luồng 2: Thu thập trải nghiệm và huấn luyện -------------
def collection_training_thread(agent: Agent, buffer: ReplayBuffer, actor: Actor, critic: Critic,
                               queue: Queue, stop_flag: Value, lock: Lock):
    """
    Luồng 2: Thu thập trải nghiệm và huấn luyện mô hình.
    - Điều khiển agent tạm dừng trong lúc thu thập dữ liệu.
    - Cập nhật tham số sau khi huấn luyện và gửi sang luồng 1.
    """
    num_traj = 32  # Số lượng trajectory để huấn luyện
    num_samples_traj = 300  # Số mẫu tối đa trong mỗi trajectory

    traj_counter = 0  # Đếm số lượng trajectory đã thu thập
    timestep_clock_counter = 0  # Đếm số timestep hiện tại
    samples_of_traj_counter = 0  # Đếm số lượng sample đã thu thập trong trajectory hiện tại

    while traj_counter < num_traj:
        with lock:
            stop_flag.value = True  # Yêu cầu luồng 1 tạm dừng việc tự động di chuyển

        if samples_of_traj_counter < num_samples_traj and not agent.S_t.isTerminalState:
            collect_sample(
                agent=agent,
                buffer=buffer,
                actor=actor,
                critic=critic,
                traj_id=traj_counter,
                tms_traj=samples_of_traj_counter,
                tms_clk=timestep_clock_counter
            )
            timestep_clock_counter += 1
            samples_of_traj_counter += 1
        else:
            traj_counter += 1
            samples_of_traj_counter = 0
            timestep_clock_counter = 0

    # Huấn luyện Actor-Critic sau khi thu thập dữ liệu
    buffer.compute_returns_and_advantages()
    new_actor_params = train_actor_critic(buffer, actor, critic)
    queue.put(new_actor_params)  # Gửi tham số mới cho luồng 1

    with lock:
        stop_flag.value = False  # Tiếp tục cho phép luồng 1 chạy


# ----------- Main -------------
def main(use_cuda=False):
    # Khởi tạo Agent và môi trường
    agt_xml_path = 'structures/agility_cassie/cassie.xml'
    agt = Agent(agt_xml_path)

    env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
    agt.add_env(env, 'cassie_env')
    agt.set_state(0)

    # Khởi tạo Replay Buffer
    buffer = ReplayBuffer(
        trajectory_size=32 * 300,
        max_size=50000,
        gamma=0.99, lam=0.95, alpha=0.6,
        file_path='env/agent/replay_buffer'
    )

    # Khởi tạo Actor và Critic
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    actor = Actor(input_size=42, output_size=30).to(device)
    critic = Critic(input_size=42).to(device)

    # Tạo các công cụ đồng bộ hóa
    queue = Queue()
    stop_flag = Value('b', False)  # Cờ dừng (True: dừng, False: chạy)
    lock = Lock()

    # Tạo và chạy các luồng
    process_autonomous = Process(target=autonomous_thread, args=(agt, queue, stop_flag, lock))
    process_training = Process(target=collection_training_thread,
                               args=(agt, buffer, actor, critic, queue, stop_flag, lock))

    process_autonomous.start()
    process_training.start()

    process_autonomous.join()
    process_training.join()


# Chạy chương trình
if __name__ == "__main__":
    main(use_cuda=True)
