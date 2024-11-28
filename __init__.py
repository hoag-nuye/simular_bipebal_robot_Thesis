import multiprocessing
import torch

def check_max_threads():
    max_threads = multiprocessing.cpu_count()  # Số luồng tối đa mà hệ thống hỗ trợ
    print(f"Số lượng luồng tối đa mà máy tính của bạn có thể chạy: {max_threads}")
    return max_threads


# Kiểm tra số luồng tối đa
# check_max_threads()
print(torch.cuda.is_available())
