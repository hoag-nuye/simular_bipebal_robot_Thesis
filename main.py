from env.mujoco_env import Environment


def main():
    # initial environment
    env_xml_path = 'structures/environment.xml'
    env = Environment(env_xml_path)

    # starting simulation
    with env.viewer:
        env.render()

if __name__ == "__main__":
    # main()
    print()


