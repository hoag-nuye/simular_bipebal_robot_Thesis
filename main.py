import mujoco
import time
import numpy as np

from env.mujoco_env import Environment
from env.agent.mujoco_agt import Agent


# Hàm lấy vị trí tiếp xúc từ left-foot

def main():
    # -------------- initial environment --------------------
    agt_xml_path = 'structures/agility_cassie/cassie.xml'
    agt = Agent(agt_xml_path)

    # -------------- Add a new agent ---------------------
    env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
    agt.add_env(env, 'cassie_env')

    # Setup real-time control
    policy_counter = 0  # Đếm số bước để cập nhật policy
    print_counter = 0  # Đếm số bước để in trạng thái
    steps_per_policy = int(1 / (40 * agt.agt_model.opt.timestep))  # Số bước cho tần số 40 Hz
    steps_per_print = int(1 / agt.agt_model.opt.timestep)  # Số bước để in mỗi giây (1 Hz)

    policy_counter = 0  # Đếm số bước để cập nhật policy
    print_counter = 0  # Đếm số bước để in trạng thái
    steps_per_policy = int(1 / (40 * agt.agt_model.opt.timestep))  # Số bước cho tần số 40 Hz
    steps_per_print = int(2 / agt.agt_model.opt.timestep)  # Số bước để in mỗi giây (1 Hz)

    time_step = 0
    time_clock = 0
    num_clock = 100
    theta_left = 0
    theta_right = 0.5
    r = 0.6
    agt.set_clock(r=r, N=num_clock, theta_left=theta_left, theta_right=theta_right)

    # ============================ SIMULAR ===================
    # starting simulation
    # Create viewer with UI options
    with mujoco.viewer.launch_passive(agt.agt_model, agt.agt_data, show_left_ui=True, show_right_ui=True) as viewer:
        # print("ACTUATORs:", agt.get_actuators_map())
        # print("SENSORs:", agt.get_sensors_map())
        #
        # # Thông tin trạng thái ban đầu
        # print("STATE:")
        # for key, value in agt.get_sensors_info().items():
        #     print(key, ': ', value)

        # ------------ RUNNING ------------------

        while viewer.is_running():
            agt.render(viewer)

            # Cập nhật policy mỗi 40 Hz
            if policy_counter % steps_per_policy == 0:
                pass
                # Tính policy mới
                # Gửi hành động tới robot

            # In thông số trạng thái mỗi giây (1 Hz)
            if print_counter % steps_per_print == 0:
                time_step += 1
                print(f"Time step: {time_step}")
                # print("STATE:")
                # for key, value in agt.get_sensors_info().items():
                #     print(key, ': ', value)

                agt.get_state(time_step)
                # for name_state, value in agt.S_t:
                #     print(f"{name_state}: {value}")
                # if time_clock < num_clock:
                #     print(f"Input: {agt.traj_input(0, 0, time_clock)}")
                #     time_clock += 1
                # else:
                #     time_clock = 0
                # for k,v in agt.atr_ctrl_map.items():
                #     print(f"{k}: {v}")
                # for k,v in agt.atr_map.items():
                #     print(f"{k}: {v}")

            # Cập nhật bộ đếm
            policy_counter += 1
            print_counter += 1


# ================== RUN MAIN =======================


if __name__ == "__main__":
    main()


