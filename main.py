import mujoco
import time

from env.mujoco_env import Environment
from env.agent.mujoco_agt import Agent

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

    # starting simulation
    # Create viewer with UI options
    with mujoco.viewer.launch_passive(agt.agt_model, agt.agt_data, show_left_ui=True, show_right_ui=True) as viewer:
        print("ACTUATORs:", agt.get_actuators_map())
        print("SENSORs:", agt.get_sensors_map())

        # Thông tin trạng thái ban đầu
        print("STATE:")
        for key, value in agt.get_sensors_info().items():
            print(key, ': ', value)

        # ------------ RUNNING ------------------
        policy_counter = 0  # Đếm số bước để cập nhật policy
        print_counter = 0  # Đếm số bước để in trạng thái
        steps_per_policy = int(1 / (40 * agt.agt_model.opt.timestep))  # Số bước cho tần số 40 Hz
        steps_per_print = int(5 / agt.agt_model.opt.timestep)  # Số bước để in mỗi giây (1 Hz)

        while viewer.is_running():
            agt.render(viewer)

            # Cập nhật policy mỗi 40 Hz
            if policy_counter % steps_per_policy == 0:
                pass
                # Tính policy mới
                # Gửi hành động tới robot

            # In thông số trạng thái mỗi giây (1 Hz)
            if print_counter % steps_per_print == 0:
                print("STATE:")
                for key, value in agt.get_sensors_info().items():
                    print(key, ': ', value)
                print("qpos:")

                print(len(agt.get_sensors_info()), len(agt.sensors_data))

            # Cập nhật bộ đếm
            policy_counter += 1
            print_counter += 1


# ================== RUN MAIN =======================


if __name__ == "__main__":
    main()


