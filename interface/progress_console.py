# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import time
import sys


def convert_second2time(times):
    # Đảm bảo đầu vào là số nguyên
    times = int(times)
    # Tính toán chi tiết từng đơn vị thời gian
    days = times // 86400
    remaining_seconds = times % 86400

    hours = remaining_seconds // 3600
    remaining_seconds %= 3600

    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60

    # Tạo chuỗi kết hợp biểu tượng
    # return f"{days:02d}d🗓 {hours:02d}h ⏱{minutes:02d}m ⏲{seconds:02d}s ⏳"

    # Tạo chuỗi kết quả
    result = []
    if days > 0:
        result.append(f"{days:02d}d")
    if hours > 0:
        result.append(f"{hours:02d}h")
    if minutes > 0:
        result.append(f"{minutes:02d}m")
    if seconds > 0 or not result:  # Luôn hiển thị giây nếu không có giá trị nào khác
        result.append(f"{seconds:02d}s ⏱ ")

    return ":".join(result)


def colorize_bar(percent):
    """
    Chuyển màu dần từ đỏ sang xanh lá cây dựa trên % tiến độ.
    """

    red = int(255 * (1 - percent))  # Màu đỏ giảm dần
    green = int(255 * percent)  # Màu xanh tăng dần
    color_code = f"\033[38;2;{red};{green};0m"  # Mã ANSI màu
    return color_code


def progress_console(total_steps, current_steps, begin_time, ncols=80):
    # Cập nhật phần trăm
    percent = current_steps / total_steps
    elapsed_time = time.time() - begin_time
    remaining_time = 0 if percent == 0 else elapsed_time / percent
    rate = 0 if elapsed_time == 0 else int(current_steps / elapsed_time)

    num_done = int(percent * ncols)
    num_left = ncols - num_done
    #  ====================== CONSOLE BEGIN ========================
    reset = "\033[0m"  # Reset màu về mặc định
    sys.stdout.write("\033[2J\033[H")  # xóa toàn bộ màn hình và đặt lai con trỏ
    sys.stdout.write(f"Loading: {colorize_bar(percent)}{'█' * num_done}{reset}{'.' * num_left} {int(percent * 100)}%\n")
    sys.stdout.write(f"Samples: {current_steps}/{total_steps}\n")
    sys.stdout.write(f"Passed: {elapsed_time}\n")
    sys.stdout.write(f"Estimate: {convert_second2time(remaining_time)} \n")
    sys.stdout.write(f"Speed: {rate:} samples/second")
    sys.stdout.flush()
    print()  # In một dòng trống sau khi hoàn thành
    # ======================== END CONSOLE ========================

# days = 10
# hours = 2
# minutes = 2
# seconds = 2
# # Tạo chuỗi kết quả
# result = []
# if days > 0:
#     result.append(f"{days:02d}d")
# if hours > 0:
#     result.append(f"{hours:02d}h")
# if minutes > 0:
#     result.append(f"{minutes:02d}m")
# if seconds > 0 or not result:  # Luôn hiển thị giây nếu không có giá trị nào khác
#     result.append(f"{seconds:02d}s ⏱")
#
# print(":".join(result))

# total = 1000
# steps = 10
# current = 0  # Bắt đầu từ 0
#
# begin_time = time.time()
# while current < total:
#     start_time = time.time()
#     current += steps
#     time.sleep(0.05)  # Giả lập công việc
#     progress_console(total_steps=total, current_steps=current, begin_time=begin_time)


# =========== TEST ===================
# import time
# import sys
#
#
# def tqdm_example(iterable, total=None, desc='', ncols=80, ascii=False):
#     # Nếu không có tổng số phần tử, tính tổng từ iterable
#     if total is None:
#         total = len(iterable)
#
#     # Thiết lập thanh tiến trình ban đầu
#     # print(f"{desc} [{'.' * ncols}]")
#     i = 0
#     while i< 100:
#
#         # Dừng lại một chút để giả lập tiến trình
#         time.sleep(0.1)
#         i+=1
#         # Tính toán phần trăm hoàn thành
#         progress = (i + 1) / total
#         num_done = int(progress * ncols)
#         num_left = ncols - num_done
#
#         # In lại thanh tiến trình (sử dụng '\r' để di chuyển con trỏ về đầu dòng)
#         sys.stdout.write("\033[F" * 2)  # Di chuyển con trỏ lên 1 dòng trước đó
#         sys.stdout.write("\033[K")  # Xóa toàn bộ dòng hiện tại
#         sys.stdout.write(f"\r{desc} [{'█' * num_done}{'.' * num_left}] {int(progress * 100-1)}%\n")
#         sys.stdout.write(f"{num_done}/{num_left}")
#         sys.stdout.flush()
#
#     print()  # In một dòng trống sau khi hoàn thành
#
#
# # Ví dụ sử dụng tqdm_custom
# items = range(100)
# tqdm_example(items, desc="Processing")
