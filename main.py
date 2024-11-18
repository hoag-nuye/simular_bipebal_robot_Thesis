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

    # starting simulation
    # Create viewer with UI options
    with mujoco.viewer.launch_passive(agt.agt_model,
                                      agt.agt_data,
                                      # key_callback=self.key_callback,
                                      show_left_ui=True,
                                      show_right_ui=True) as viewer:

        print("ACTUATORs:", agt.get_actuators_map())
        print("SENSORs:", agt.get_sensors_map())

        # ------------ CONSOLE STATE ------------------
        print("STATE:")
        for key, value in agt.get_sensors_info().items():
            print(key, ': ', value)

        # ------------ RUNNING ------------------
        while viewer.is_running():
            agt.render(viewer)

            # chờ 2 giây
            # time.sleep(2)
            print("STATE:")
            status_line = " | ".join([f"{key}: {value:.2f}" for key, value in agt.get_sensors_info().items()])
            print(f"\rSTATE: {status_line}", end="")

# ================== RUN MAIN =======================


if __name__ == "__main__":
    main()
    print()


