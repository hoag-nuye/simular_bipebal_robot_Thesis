# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import mujoco

class Environment:
    def __init__(self, env_xml, env_data):
        """
        Khởi tạo một Environment từ XML và `agent_data`.
        env_xml: Đường dẫn tới tệp XML của env.
        agt_data: Dữ liệu của môi trường MuJoCo.
        """
        self.env_xml = env_xml
        self.env_data = env_data





