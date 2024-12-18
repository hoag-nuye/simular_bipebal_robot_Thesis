# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import os
import multiprocessing

from env.agent.mujoco_agt import Agent
from env.mujoco_env import Environment
from interface.mujoco_viewer import mujoco_viewer_process, mujoco_viewer_process_begin, mujoco_viewer_process_external_state
from interface.plot_param import plot_param_process

# Số lõi logic (logical cores)
logical_cores = os.cpu_count()
print(f"Số logical cores: {logical_cores}")

# Số lõi vật lý (physical cores)
physical_cores = multiprocessing.cpu_count()
print(f"Số physical cores: {physical_cores}")

print(multiprocessing)

# -------------- initial environment --------------------
agt_xml_path = 'structures/agility_cassie/cassie.xml'
agt = Agent(agt_xml_path)

# -------------- Add a new agent ---------------------
env = Environment('structures/agility_cassie/environment.xml', agt.agt_data)
agt.add_env(env, 'cassie_env')
agt.set_state(0)  # Khởi tạo giá trị ban đầu

# mujoco_viewer_process_begin(agt)
# plot_param_process("models/param/")
# mujoco_viewer_process(agt=agt, policy_freq=40, pd_control_freq=2000, use_cuda=True)
mujoco_viewer_process_external_state(agt=agt, policy_freq=40, pd_control_freq=2000, use_cuda=True)
