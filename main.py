from env.mujoco_env import Environment

# initial environment
env_xml_path = 'structures/environment.xml'
env = Environment(env_xml_path)

with env.viewer:
    env.render()


