class Agent:
    def __init__(self, agent_xml, env_data):
        """
        Khởi tạo một Agent từ XML và môi trường `env_data`.
        agent_xml: Đường dẫn tới tệp XML của agent.
        env_data: Dữ liệu của môi trường MuJoCo.
        """
        self.agent_xml = agent_xml
        self.env_data = env_data
        # Lưu dữ liệu state hiện tại của agent
        self.agent_pos = None
        self.agent_vel = None
        # Lưu trữ tập trải nghiệm
        self.trajectory_buffer = []






