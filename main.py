from env.mujoco_env import Environment
from env.mujoco_agt import Agent


def main():
    # -------------- initial environment --------------------
    env_xml_path = 'structures/agility_cassie/environment.xml'
    env = Environment(env_xml_path)

    # -------------- Add a new agent ---------------------
    agility_cassie_agent = Agent('structures/agility_cassie/cassie.xml', env.env_data)
    env.add_agent(agility_cassie_agent, 'cassie')

    # starting simulation
    with env.viewer:
        env.render()

if __name__ == "__main__":
    main()
    print()


