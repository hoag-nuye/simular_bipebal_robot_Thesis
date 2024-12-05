import time
import copy
import random
from functools import partial

from multiprocessing import Pool
from env.mujoco_env import Environment

from env.agent.mujoco_agt import Agent
from env.agent.buffer import ReplayBuffer, ReplayCache
from interface.interface_main import trajectory_collection, train
from interface.plot_param import plot_param_process


# Hàm được gọi bởi mỗi tiến trình
# Hàm được gọi trong Pool
def process_trajectory_collection(func_with_args):
    try:
        func_with_args()  # Thu thập trajectory
        return func_with_args.keywords["buffer"].get_samples()  # Trả về dữ liệu từ ReplayCache
    except Exception as e:
        print(f"Error: {e}")
        return None  # Trả về None nếu có lỗi

"""
============================================================
-------------------------- MAIN ----------------------------
============================================================
"""

# ================== RUN MAIN =======================
if __name__ == "__main__":
    # Bật/tắt CUDA bằng cách thay đổi giá trị `use_cuda`
    use_cuda = True

    # -------------- initial environment --------------------
    agt_xml_path = 'structures/agility_cassie/cassie.xml'
    agt = Agent(agt_xml_path)

    # -------------- Add a new agent ---------------------
    env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
    agt.add_env(env, 'cassie_env')
    agt.set_state(0)  # Khởi tạo giá trị ban đầu

    # ======================== THU THẬP 32 TRAJECTORY =======================
    # Tạo tham số cho lần thua thập
    num_traj_train = 32  # Số trajectory cần thu thập  cho mỗi lần train
    num_samples_traj = 300  # Số samples tối đa trong 1 trajectory

    # ----------------- REPLAY BUFFER --------------------
    dir_file = 'env/agent/'
    Buffer = ReplayBuffer(trajectory_size=num_traj_train * num_samples_traj,
                          max_size=50000,
                          gamma=0.99, lam=0.95, alpha=0.6,
                          file_path=f"{dir_file}replay_buffer")

    # ------------------ SETUP TIME CONTROL --------------------
    agt.agt_model.opt.timestep = 0.0005  # Set timestep to achieve 2000Hz
    # Định nghĩa tần số
    policy_freq = 40  # Tần số chính sách (Hz) Trong 1s thu thập được 40 trạng thái
    pd_control_freq = 2000  # Tần số bộ điều khiển PD (Hz)
    steps_per_policy = pd_control_freq // policy_freq  # Số bước PD trong mỗi bước chính sách

    # ================= CHẠY ĐA LUỒNG ================
    num_processes = 4
    start_time = time.time()
    # Số lượng trajectory cần thu thập cho mỗi luồng
    num_traj = 8  # Số trajectory cần thu thập cho mỗi luồng
    num_samples_traj = 300  # Số samples tối đa trong 1 trajectory

    # Tạo tham số cho mô hình Actor và Critic
    traj_input_size = 42
    traj_output_size = 60
    # Tạo replay_cache
    buffer_path_dir = 'env/agent/'
    param_path_dir = 'models/param/'

    for num_train in range(4):  # 2 số lần thu thập 32 trajectory

        # Create clock for agent and control
        num_clock = 100
        theta_left = 0
        theta_right = 0.5
        r = 0.6
        agt.set_clock(r=r, N=num_clock, theta_left=theta_left, theta_right=theta_right)
        agt.x_des_vel = random.uniform(-1.5, 1.5)  # Random từ -1.5 đến 1.5 m/s
        agt.y_des_vel = random.uniform(-1.0, 1.0)  # Random từ -1.0 đến 1.0 m/s

        # ----------------- THU THẬP TRẢI NGHIỆM -----------------

        # Chuẩn bị tham số cho Pool
        tasks = []
        for process_idx in range(num_processes):
            agt_copy = copy.deepcopy(agt)
            replay_cache = ReplayCache(trajectory_size=num_traj * num_samples_traj)

            traj_id_start = int(num_processes*num_traj*num_train) + num_traj*process_idx

            func_with_args = partial(trajectory_collection,
                                     traj_id_start=traj_id_start,
                                     num_traj=num_traj,
                                     num_samples_traj=num_samples_traj,
                                     num_timestep_clock=num_clock,
                                     num_steps_per_policy=steps_per_policy,
                                     agt=agt_copy,
                                     traj_input_size=traj_input_size,
                                     traj_output_size=traj_output_size,
                                     param_path_dir=param_path_dir,
                                     use_cuda=use_cuda,
                                     buffer=replay_cache)

            # Thay thế việc lưu func_with_args vào danh sách khác nếu cần
            tasks.append(func_with_args)

        # Sử dụng Pool để thực thi song song
        with Pool(processes=num_processes) as pool:
            # Thực thi hàm process_trajectory_collection trên tất cả func_with_args
            results = pool.map(process_trajectory_collection, tasks)

            # Kết hợp kết quả vào ReplayBuffer
            for result in results:
                if result is not None:
                    agt.total_samples += len(next(iter(result.values())))
                    Buffer.append_from_buffer(result)

            # # Kiểm tra dữ liệu trong Buffer
            # for k, v in Buffer.get_samples().items():
            #     print(f"{k}: {len(v)}")

        Buffer.compute_returns_and_advantages()
        Buffer.update_td_errors()
        Buffer.save_to_pkl()
        Buffer.reset()

        data_batch = Buffer.sample_batch(batch_size=32)
        # for k, v in data_batch.items():
        #     print(f"Size of {k} is: {len(v)}, shape:{v.shape} ({type(v)})")
        """ 
        ============================================================
        QUÁ TRÌNH TRAIN DIỄN RA SAU 1 LẦN THU THẬP ĐỦ SỐ LƯỢNG BATCH
        ============================================================
        """
        train(training_id=num_train,
              data_batch=data_batch,
              epochs=4,
              learning_rate=0.0001,
              output_size=traj_output_size,
              path_dir="models/param/",
              use_cuda=use_cuda)

    end_time = time.time()
    execution_time = end_time - start_time
    print(execution_time)
    print(agt.total_samples)
    plot_param_process(path_dir=param_path_dir)

""" 
============================================================
QUÁ TRÌNH TRAIN DIỄN RA SAU 1 LẦN THU THẬP ĐỦ SỐ LƯỢNG BATCH
============================================================
"""
#
# # train(training_id=train_counter,
# #       data_batch=data_batch,
# #       epochs=4,
# #       learning_rate=0.0001,
# #       output_size=traj_output_size,
# #       path_dir="models/param",
# #       use_cuda=use_cuda)
# # train_counter += 1
#
# # Kết thúc đo thời gian
# end_time = time.time()
# # Tính toán và in ra thời gian chạy
# execution_time = end_time - start_time
# print(f"Thời gian thực hiện COLLECTION & TRAIN: {execution_time} s")
