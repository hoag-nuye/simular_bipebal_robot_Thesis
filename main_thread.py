# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import copy
import time
import torch
import keyboard
import numpy as np
from functools import partial
from signal import signal, SIGINT
from multiprocessing import Pool, Process, Lock, Value

from env.agent.mujoco_agt import Agent
from env.mujoco_env import Environment
from interface.plot_param import plot_param_process
from env.agent.buffer import ReplayBuffer, ReplayCache
from interface.progress_console import progress_console, train_progress_console
from interface.mujoco_viewer import mujoco_viewer_process
from interface.interface_main import trajectory_collection, train

"""
************************************************************
---------------------- SUPPORT MAIN ------------------------
************************************************************
"""


# Hàm sẽ được gọi khi nhận tín hiệu ngắt
def initializer(share_var, share_lock):
    import global_value as glv
    glv.g_lock = share_lock
    glv.g_shared_var = share_var
    signal(SIGINT, lambda: None)  # Dừng tiến trình khi có yêu cầu


# Hàm được gọi bởi mỗi tiến trình
# Hàm được gọi trong Pool
def process_trajectory_collection(task):
    try:
        task()  # Thu thập trajectory
        return task.keywords["buffer"]  # Trả về dữ liệu từ ReplayCache
    except Exception as e:
        print(f"Error: {e}")
        return None  # Trả về None nếu có lỗi


"""
************************************************************
-------------------------- MAIN ----------------------------
************************************************************
"""

# ================== RUN MAIN =======================
if __name__ == "__main__":

    """ 
    ============================================================
    ************* CÁC HÀM HIỂN THỊ KHI NHẤN PHÍM TẮT ***********
    ------------------------------------------------------------
    """

    mujoco_process = None
    plot_process = None

    # Hàm hiển thị mô phỏng
    def togger_mujoco(is_enable):
        global mujoco_process
        if mujoco_process is None:
            mujoco_process = Process(target=mujoco_viewer_process, args=(is_enable, ))
            mujoco_process.start()
        elif mujoco_process is not None:
            if mujoco_process.is_alive():
                print("\nMujoco sim exited!")
                mujoco_process.terminate()
                mujoco_process.join()  # Đợi process kết thúc hoàn toàn và tài nguyên được giải phóng
                mujoco_process = None
            else:
                print("\nMujoco sim exited!")
                mujoco_process.terminate()
                mujoco_process.join()  # Đợi process kết thúc hoàn toàn và tài nguyên được giải phóng
                mujoco_process = None
        else:
            mujoco_process = None

    # Hàm hiển thị biểu đồ tham số
    def togger_plot(_path_dir):
        global plot_process
        if plot_process is None and (_path_dir is not None):
            plot_process = Process(target=plot_param_process, args=(_path_dir,))
            plot_process.start()
        elif plot_process is not None:
            if plot_process.is_alive():
                print("\nPlotting exited!")
                plot_process.terminate()
                plot_process.join()  # Đợi process kết thúc hoàn toàn
                plot_process = None
            else:
                print("\nPlotting exited!")
                plot_process.terminate()
                plot_process.join()  # Đợi process kết thúc hoàn toàn
                plot_process = None
        else:
            plot_process = None


    """ 
    ============================================================
    *********** CHƯƠNG TRÌNH THU THẬP VÀ HUẤN LUYỆN ************
    ------------------------------------------------------------
    """

    """
    =======================================================
    --------- KHỞI TẠO DỮ LIỆU CHO CHƯƠNG TRÌNH -----------
    =======================================================
    """

    # ************** THAM SỐ CHO MUJOCO ***************
    agt_xml_path = 'structures/agility_cassie/cassie.xml'
    agt = Agent(agt_xml_path)

    # Thêm thông tin môi trường
    env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
    agt.add_env(env, 'cassie_env')

    # Set timesteps caculator of mujoco
    agt.agt_model.opt.timestep = 0.0005

    # ************** THAM SỐ CHO THU THẬP VÀ TRAIN **************
    # Phần cứng sử dụng
    use_cuda = True
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    # print(device)
    # ----------------- REPLAY BUFFER ---------------
    # Số trajectory cần thu thập  cho mỗi lần train
    num_traj_train = 32

    # Số samples tối đa trong 1 trajectory
    num_samples_traj = 300
    dir_file = 'env/agent/'
    Buffer = ReplayBuffer(trajectory_size=num_traj_train * num_samples_traj,
                          max_size=50000,
                          gamma=0.99, lam=0.95, alpha=0.6,
                          file_path=f"{dir_file}replay_buffer")

    # ----------------- THAM SỐ ĐIỀU KHIỂN AGENT ---------------
    # Tần số chính sách (Hz) Trong 1s thu thập được 40 trạng thái
    policy_freq = 40

    # Tần số bộ điều khiển PD (Hz)
    pd_control_freq = 2000

    # Số bước PD trong mỗi bước chính sách
    steps_per_policy = pd_control_freq // policy_freq

    # Số luông thu thập
    num_processes = 4
    # Số lượng trajectory cần thu thập cho mỗi luồng
    max_sample_collect = 50000
    # Số sample thu thập tối đa
    num_samples_traj = 300  # Số samples tối đa trong 1 trajectory
    num_samples = 50000000  # số lượng sample cần thu thập

    # Tạo tham số cho mô hình Actor và Critic
    traj_input_size = 58
    traj_output_size = 60
    # Tạo replay_cache
    buffer_path_dir = 'env/agent/'

    # Tạo tham số huấn luyện (bằng với số iteration)
    max_training_id = int(num_samples / (num_samples_traj * num_traj_train))  # số lần train tối đa

    # ----------------- THAM SỐ CHO CÁC LUỒNG DỮ LIỆU -----------------
    # Phím tắt cho 2 luồng hiển thị
    param_path_dir = 'models/param/'

    # Luồng hiển thị mô phỏng
    is_enable_sim = True
    keyboard.add_hotkey('v', togger_mujoco, args=(is_enable_sim,))

    # Luồng hiển thị biểu đồ
    keyboard.add_hotkey('b', togger_plot, args=(param_path_dir,))

    """ 
    ============================================================
    ----------- QUÁ TRÌNH THU THẬP DỮ LIỆU DIỄN RA -------------
    ============================================================
    """

    # ********** KHỞI TẠO BIẾN CHO THU THẬP ***********
    # ============== KHỞI TẠO TẦN SỐ NHỊP BƯỚC ============
    # Tạo tín hiệu điều khiển agent
    agt.x_des_vel = 1.35  # Random từ -1.5 đến 1.5 m/s
    agt.y_des_vel = 0  # Random từ -1.0 đến 1.0 m/s
    range_steps = 0.6  # (m)
    cycle_time_steps = range_steps/agt.x_des_vel  # thời gian hoàn thành 1 chu kì bước với vận tốc cho trc
    num_clock = int(cycle_time_steps/(1/policy_freq))
    theta_left = 0
    theta_right = 0.55
    r = 0.6  # Độ dài pha nhấc chân
    a_i = 0.5  # Pha tại t = 0
    agt.set_clock(r=r, N=num_clock, theta_left=theta_left, theta_right=theta_right, a_i=a_i)
    # agt.x_des_vel = random.uniform(-1.5, 1.5)  # Random từ -1.5 đến 1.5 m/s
    # agt.y_des_vel = random.uniform(-1.0, 1.0)  # Random từ -1.0 đến 1.0 m/s

    # agt.x_des_vel = 0
    # agt.y_des_vel = 0

    # dữ nguyên robot ở thẳng đứng
    agt.quat_des = np.array([1, 0, 0, 0])
    # Tạo biến dừng cho chương trình
    is_done = False

    start_time = time.time()

    # Bắt ngoại lệ để dừng tiến trình trong terminal
    try:
        #  Thu thập kết thúc khi thu thập đủ 150,000,000 sample
        while agt.total_samples < num_samples:
            is_enable_sim = True
            # ============== HIỂN THỊ THỜI GIAN CHẠY TIẾN TRÌNH ================
            progress_console(total_steps=num_samples,
                             current_steps=agt.total_samples,
                             begin_time=start_time)



            # ============== THU THẬP TRẢI NGHIỆM ==============
            # Chuẩn bị tham số cho Pool
            lock = Lock()
            shared_var = Value('i', 0)  # Biến chia sẻ
            get_results = []  # Lấy kết quả ra khỏi đa luồng tránh lỗi dữ liệu
            tasks = []
            current_max_traj_id = 0

            # Tạo thông tin cho từng luồng
            for process_idx in range(num_processes):
                agt_copy = copy.deepcopy(agt)
                replay_cache = ReplayCache()

                func_with_args = partial(trajectory_collection,
                                         max_sample_collect=max_sample_collect,
                                         num_samples_traj=num_samples_traj,
                                         num_timestep_clock=num_clock,
                                         num_steps_per_policy=steps_per_policy,
                                         agt=agt_copy,
                                         traj_input_size=traj_input_size,
                                         traj_output_size=traj_output_size,
                                         param_path_dir=param_path_dir,
                                         device=device,
                                         buffer=replay_cache)

                # Thay thế việc lưu func_with_args vào danh sách khác nếu cần
                tasks.append(func_with_args)
            try:

                # ********** THU THẬP DỮ LIỆU *************
                # Sử dụng Pool để thực thi song song
                with Pool(processes=num_processes, initializer=initializer, initargs=(shared_var, lock)) as pool:
                    # Thực thi hàm process_trajectory_collection trên tất cả func_with_args
                    results = pool.map(process_trajectory_collection, tasks)
                    # Kết hợp kết quả vào ReplayBuffer
                    for result in results:
                        if result is not None:
                            # Thiết lập lại id traj
                            # print(current_max_traj_id)
                            current_max_traj_id = result.sget_range_trajectory(current_max_traj_id)

                            # Lấy dữ liệu data của relay buffer
                            get_results.append(result.get_samples())

            except KeyboardInterrupt:
                pool.terminate()
                pool.join()

            # Lấy dữ liệu
            for get_result in get_results:
                # tính tổng sample đã thu thập
                agt.total_samples += len(next(iter(get_result.values())))

                # Gộp dữ liệu từ các luồng
                Buffer.append_from_buffer(get_result)

            # Tạo mini_batch cho quá trình huấn luyện
            mini_batch = Buffer.sample_batch(batch_size=32)
            mini_batch_size = len(mini_batch)

            # Xóa toàn bộ dữ liệu có trong Buffer (giảm bộ nhớ ram)
            Buffer.reset()
            # for i in range(mini_batch[0]["rewards"].shape[0]):
            #     print(mini_batch[0]["rewards"][i,:, :])
            # break
            """ 
            ============================================================
            ---------------- QUÁ TRÌNH HUẤN LUYỆN DIỄN RA --------------
            ============================================================
            """
            # ********** KHỞI TẠO BIẾN CHO HUẤN LUYỆN ************
            # Khởi tạo epoch : số lần nhìn thấy toàn bộ dữ liệu
            epoch = 4

            # Số lần train qua 1 batch_size
            iters_passed = 0

            # NOTE: mini_batch được tạo ra từ Buffer
            # NOTE: Biến max_training_id chính là số iteration (số lần nhìn batch in mini_batch)

            # Tham số học
            actor_learning_rate = 0.0001
            critic_learning_rate = 0.0001

            clip_value = 1.0  # Tránh bùng nổ gradient

            # Khả năng khám phá (Không sử dụng)
            entropy_weight = 0.02

            # Ngưỡng cho thuật toán PPOClip
            epsilon = 0.2

            # Địa chỉ lưu dữ liệu huấn luyện
            path_dir = "models/param/"

            # Phần cứng sử dụng
            device = device

            # Dừng việc sử dụng view mujoco trong lúc huấn luyện
            is_enable_sim = False
            # ********** HUẤN LUYỆN ************
            # Lặp qua từng epoch
            for _ in range(epoch):
                start_time_epoch = time.time()
                for idx, batch in enumerate(mini_batch):
                    iters_passed += idx
                    actor_loss, critic_loss, mean_reward = \
                        train(agt=agt,
                              iters_passed=iters_passed,
                              data_batch=batch,
                              # is_save=True if _ == epoch-1 else False,
                              is_save=True,
                              epochs=epoch,
                              actor_learning_rate=actor_learning_rate,
                              critic_learning_rate=critic_learning_rate,
                              clip_value=clip_value,
                              entropy_weight=entropy_weight,
                              epsilon=epsilon,
                              output_size=traj_output_size,
                              path_dir=path_dir,
                              device=device)

                    # ============== HIỂN THỊ THỜI GIAN CHẠY TIẾN TRÌNH ================
                    train_progress_console(total_steps=mini_batch_size,
                                           current_steps=idx + 1,
                                           begin_time=start_time_epoch,
                                           current_epoch=_ + 1,
                                           total_epoch=epoch,
                                           actor_loss=actor_loss,
                                           critic_loss=critic_loss,
                                           mean_reward=mean_reward)

        """         
        ============================================================
        ---------------- QUÁ TRÌNH XỬ LÝ TÍN HIỆU NGẮT -------------
        ============================================================
        """
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
