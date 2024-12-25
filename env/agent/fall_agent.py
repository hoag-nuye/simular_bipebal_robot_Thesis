from collections import deque


# Class kiểm tra xem agent có đang bị ngã hay không
class FallDetector:
    def __init__(self, n, min_force_threshold=10, max_tilt_threshold=30):
        """
        Khởi tạo bộ phát hiện ngã:
        - n: Số lượng giá trị gần nhất cần lưu.
        - min_force_threshold: Ngưỡng lực tối thiểu để kiểm tra ngã.
        - max_tilt_threshold: Ngưỡng góc nghiêng tối đa để kiểm tra ngã (độ).
        """
        self.n = n
        self.min_force_threshold = min_force_threshold
        self.max_tilt_threshold = max_tilt_threshold
        self.left_force_queue = deque(maxlen=n)
        self.right_force_queue = deque(maxlen=n)
        self.tilt_angle_queue = deque(maxlen=n)

    def update_data(self, left_fz, right_fz, tilt_angle):
        """
        Cập nhật giá trị lực và góc nghiêng cho hàng đợi.
        """
        self.left_force_queue.append(left_fz)
        self.right_force_queue.append(right_fz)
        self.tilt_angle_queue.append(tilt_angle)

    def reset_data(self):
        self.left_force_queue = deque(maxlen=self.n)
        self.right_force_queue = deque(maxlen=self.n)
        self.tilt_angle_queue = deque(maxlen=self.n)

    def is_fallen(self):
        """
        Kiểm tra robot có bị ngã không.
        Điều kiện:
        - 80% giá trị lực trong n gần nhất của cả 2 chân nhỏ hơn ngưỡng lực tối thiểu.
        - 80% giá trị góc nghiêng trong n gần nhất vượt quá ngưỡng góc nghiêng tối đa.
        """
        if len(self.tilt_angle_queue) >= 1:
            x = self.tilt_angle_queue[-1]
            # print(x)
            if x >= self.max_tilt_threshold:
                # print("NGÃ")
                return True

        if len(self.tilt_angle_queue) == self.n:
            min_force_count = int(0.8 * self.n)  # Số giá trị cần thiết để thỏa mãn điều kiện lực


            # Kiểm tra số lần lực nhỏ hơn ngưỡng ở mỗi chân
            left_low_force_count = sum(abs(fz) < self.min_force_threshold for fz in self.left_force_queue)
            right_low_force_count = sum(abs(fz) < self.min_force_threshold for fz in self.right_force_queue)

            # Robot được coi là ngã nếu cả hai điều kiện đều thỏa mãn
            if (left_low_force_count >= min_force_count and
                right_low_force_count >= min_force_count):
                return True

        return False

# ============== TEST THỬ =================
# # Tạo bộ phát hiện ngã với n=10, ngưỡng lực tối thiểu là 5, ngưỡng góc nghiêng tối đa là 15 độ
# fall_detector = FallDetector(n=10, min_force_threshold=5, max_tilt_threshold=15)
#
# # Dữ liệu mô phỏng (lực và góc nghiêng)
# data = [
#     (-2, -3, 10), (-4, -5, 20), (0, 0, 25), (-1, -2, 30),
#     (-3, -4, 12), (-1, -1, 18), (-2, -3, 5), (-3, -3, 22),
#     (-2, -2, 35), (-1, -1, 40)
# ]
#
# for left_fz, right_fz, tilt_angle in data:
#     fall_detector.update_data(left_fz, right_fz, tilt_angle)
#     if fall_detector.is_fallen():
#         print("Robot bị ngã!")
#         break
# else:
#     print("Robot vẫn đang đứng!")
