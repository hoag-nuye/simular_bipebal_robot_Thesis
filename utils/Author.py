# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================


import os
from datetime import datetime

# Nội dung comment mới
author_comment = (
    "# === Information ================================================================\n"
    "#  @author:       Hoàng Nguyên\n"
    "#  Email:         nguyen80o08nguyen@gmail.com\n"
    "#  Github:        hoag.nuye\n"
    f"#  Created Date:  {datetime.now().strftime('%Y-%m-%d')}\n"
    "# === Information ================================================================\n\n"
)

# Đường dẫn đến project của bạn
project_path = "../"


def add_or_update_author_comment(file_path):
    with open(file_path, "r+", encoding="utf-8") as file:
        lines = file.readlines()

        # Tìm vị trí của comment cũ
        start_index, end_index = -1, -1
        for i, line in enumerate(lines):
            if line.strip() == "# === Information ================================================================":
                if start_index == -1:
                    start_index = i
                else:
                    end_index = i
                    break

        # Xóa comment cũ nếu tồn tại
        if start_index != -1 and end_index != -1:
            del lines[start_index:end_index + 1]

        # Thêm comment mới vào đầu file
        file.seek(0)
        file.write(author_comment + "".join(lines))
        file.truncate()


def update_project_comments(path):
    for root, _, files in os.walk(path):
        # Bỏ qua thư mục .venv
        if ".venv" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                add_or_update_author_comment(os.path.join(root, file))


if __name__ == "__main__":
    update_project_comments(project_path)
    print("Đã cập nhật hoặc thêm comment tác giả vào tất cả các file .py.")
