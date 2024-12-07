import copy
import random
import time
from signal import signal, SIGINT
import sys
import keyboard
from functools import partial
from multiprocessing import Pool, Process

from env.agent.buffer import ReplayBuffer, ReplayCache
from env.agent.mujoco_agt import Agent
from env.mujoco_env import Environment
from interface.interface_main import trajectory_collection, train
from interface.plot_param import plot_param_process
from interface.mujoco_viewer import mujoco_viewer_process
from interface.progress_console import progress_console


# ==================== XỬ LÝ TÍN HIỆU NGẮT =====================
# Hàm sẽ được gọi khi nhận tín hiệu ngắt
def initializer():
    signal(SIGINT, lambda: None)


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
==================================
-------------------------- MAIN ----------------------------
============================================================
"""

# ================== RUN MAIN =======================
if __name__ == "__main__":
    """
    ========================== LUỒNG HIỂN THỊ ==========================
    """
    mujoco_process = None
    plot_process = None

    def togger_mujoco(agent):
        global mujoco_process
        if mujoco_process is None and (agent is not None):
            mujoco_process = Process(target=mujoco_viewer_process, args=(agent,))
            mujoco_process.start()
        elif mujoco_process is not None:
            if mujoco_process.is_alive():
                print("Mujoco simulation exited!")
                mujoco_process.terminate()
                mujoco_process.join()  # Đợi process kết thúc hoàn toàn và tài nguyên được giải phóng
                mujoco_process = None
            else:
                print("Mujoco simulation exited!")
                mujoco_process.terminate()
                mujoco_process.join()  # Đợi process kết thúc hoàn toàn và tài nguyên được giải phóng
                mujoco_process = None
        else:
            mujoco_process = None

    def togger_plot(path_dir):
        global plot_process
        if plot_process is None and (path_dir is not None):
            plot_process = Process(target=plot_param_process, args=(path_dir, ))
            plot_process.start()
        elif plot_process is not None:
            if plot_process.is_alive():
                print("Plotting of parameters exited!")
                plot_process.terminate()
                plot_process.join()  # Đợi process kết thúc hoàn toàn
                plot_process = None
            else:
                print("Plotting of parameters exited!")
                plot_process.terminate()
                plot_process.join()  # Đợi process kết thúc hoàn toàn
                plot_process = None
        else:
            plot_process = None

    """
    ========================== LUỒNG THU THẬP VÀ HUẤN LUYỆN AGENT ==========================
    """
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
    # ----- 2 luồng hiển thị --------
    param_path_dir = 'models/param/'
    keyboard.add_hotkey('v', togger_mujoco, args=(agt,))
    keyboard.add_hotkey('b', togger_plot, args=(param_path_dir,))

    # Số luông thu thập
    num_processes = 4
    # Số lượng trajectory cần thu thập cho mỗi luồng
    num_traj = 8  # Số trajectory cần thu thập cho mỗi luồng
    num_samples_traj = 300  # Số samples tối đa trong 1 trajectory
    num_samples = 150000000  # số lượng sample cần thu thập
    total_sample_counter = 0  # Đếm số lượng sample đã được thu thập để dừng train
    train_counter = 0  # Đếm số lần train (thu thập 32 trajectory mỗi lần)

    # Tạo tham số cho mô hình Actor và Critic
    traj_input_size = 42
    traj_output_size = 60
    # Tạo replay_cache
    buffer_path_dir = 'env/agent/'

    '''
    ======================================================
    ======================= START ========================
    ======================================================
    '''
    start_time = time.time()
    # Bắt ngoại lệ để dừng tiến trình trong terminal
    try:
        while agt.total_samples < num_samples:
            # ============== HIỂN THỊ TIẾN TRÌNH TRÊN CONSOLE ================
            progress_console(total_steps=num_samples,
                             current_steps=agt.total_samples,
                             begin_time=start_time)
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

                traj_id_start = int(num_processes * num_traj * train_counter) + num_traj * process_idx

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
            try:
                # Sử dụng Pool để thực thi song song
                with Pool(processes=num_processes, initializer=initializer) as pool:
                    # Thực thi hàm process_trajectory_collection trên tất cả func_with_args
                    results = pool.map(process_trajectory_collection, tasks)

                    # Kết hợp kết quả vào ReplayBuffer
                    for result in results:
                        if result is not None:
                            agt.total_samples += len(next(iter(result.values())))
                            Buffer.append_from_buffer(result)
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
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

            train(training_id=train_counter,
                  data_batch=data_batch,
                  epochs=4,
                  learning_rate=0.0001,
                  output_size=traj_output_size,
                  path_dir="models/param/",
                  use_cuda=use_cuda)
            train_counter += 1  # Tăng biến đếm train lên 1

    except KeyboardInterrupt:
        print("Chương trình đã bị dừng bởi người dùng.")
        # Dừng các tiến trình con
        if mujoco_process is not None:
            if mujoco_process.is_alive():
                mujoco_process.terminate()
                mujoco_process.join()  # Đợi process kết thúc hoàn toàn và tài nguyên được giải phóng
                mujoco_process = None
            else:
                mujoco_process.terminate()
                mujoco_process.join()  # Đợi process kết thúc hoàn toàn và tài nguyên được giải phóng
                mujoco_process = None
        if plot_process is not None:
            if plot_process.is_alive():
                plot_process.terminate()
                plot_process.join()  # Đợi process kết thúc hoàn toàn
                plot_process = None
            else:
                plot_process.terminate()
                plot_process.join()  # Đợi process kết thúc hoàn toàn
                plot_process = None
        print("Tất cả tiến trình đã được dừng.")

    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(execution_time)
    # print(agt.total_samples)

""" 
============================================================
QUÁ TRÌNH TRAIN DIỄN RA SAU 1 LẦN THU THẬP ĐỦ SỐ LƯỢNG BATCH
============================================================
"""
