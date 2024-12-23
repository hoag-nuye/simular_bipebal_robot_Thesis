"""
************************************************************
-------------------------- VARIABLE ------------------------


************************************************************
"""
# BIẾN DÙNG CHUNG CHO CÁC LUỒNG
global g_shared_var  # Biến đếm sample
global g_lock  # Biến khóa khi xử lý g_shared_var
global g_history_console  # Biến lưu lại những lần in trước


# def my_function():
#     if not hasattr(my_function, "counter"):
#         my_function.counter = 0  # Khởi tạo thuộc tính nếu chưa có
#     my_function.counter += 1
#     print(f"Counter hiện tại: {my_function.counter}")
#
#
# my_function()  # Output: Counter hiện tại: 1
# my_function()  # Output: Counter hiện tại: 2
"""
************************************************************
-------------------------- TEST ----------------------------
************************************************************
"""
# import numpy as np
# len_batch = 3
# unique_trajectories = [1,2,3,4,2,2,4,5,6,7,8,9,0,3,5,6]
#
#
# print(range(0, len(unique_trajectories), len_batch))
# print([unique_trajectories[i:i+len_batch] for i in range(0, 16, 3)])
# print([i for i in range(0, 16, 3)])

# buffer = {
#     "trajectory_ids": [1, 1, 2, 2, 3, 3, 4, 4],
#     "states": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]]
# }
# # value arr[1, 1, 2, 2, 3, 3, 4, 4]
# # index arr[0, 1, 2, 3, 4, 5, 6, 7]
# # index arr[0,       3,       6, 7] np.array()[[0,3,6,7]]
# # index arr[0,       1,       2, 3]
# # index arr[         1           3]            [[0,3,6,7]][[1,3]]
# # =>    arr[         2           4]
# # => Có thể lọc n lần từ mảng ban đầu theo chỉ số
# print(np.array(buffer["trajectory_ids"])[[0,3,6,7]][[1,3]])
#
# len_batch = 5
# buffer = {

#     "trajectory_ids": [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15,
#                        16, 16, 17, 18, 18, 33, 33, 34, 35, 36, 37, 38, 38, 39, 40, 41, 41, 42, 50, 51, 52, 53, 54, 55,
#                        55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
#     "states": [[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0], [1.1], [1.2], [1.3], [1.4], [1.5],
#                [1.6], [1.7], [1.8], [1.9], [2.0], [2.1], [2.2], [2.3], [2.4], [2.5], [2.6], [2.7], [2.8], [2.9], [3.0],
#                [3.1], [3.2], [3.3], [3.4], [3.5], [3.6], [3.7], [3.8], [3.9], [4.0], [4.1], [4.2], [4.3], [4.4], [4.5],
#                [4.6], [4.7], [4.8], [4.9], [5.0], [5.1], [5.2], [5.3], [5.4], [5.5], [5.6], [5.7], [5.8], [5.9], [6.0],
#                [6.1], [6.2], [6.3], [6.4], [6.5], [6.6], [6.7], [6.8], [6.9]],
#     "actions": [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18],
#                 [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35],
#                 [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52],
#                 [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69]]
# }
# mini_batch = []
#
# # Tạo các nhóm tương ứng cho trajectory_ids với chỉ mục
#
#
# trajectory_ids = buffer["trajectory_ids"]
#
# # Lấy các giá trị duy nhất từ trajectory_ids và random hóa
# unique_trajectories = np.unique(trajectory_ids)
# # np.random.shuffle(unique_trajectories)  # Xáo trộn các giá trị duy nhất
#
# # Tạo danh sách chứa các nhóm với mỗi nhóm có len_batch giá trị duy nhất
# grouped_trajectories = [unique_trajectories[i:i + len_batch].tolist() for i in
#                         range(0, len(unique_trajectories), len_batch)]
#
# batches_indices = []
# for group in grouped_trajectories:
#     group_indices = np.where(np.isin(trajectory_ids, group))[0]  # Chỉ mục tương ứng với group
#     batches_indices.append(group_indices)

# print(grouped_trajectories)
# print(batches_indices)

# for batch_indices in batches_indices:
#     batch = {key: np.array([]) for key in buffer.keys()}
#     for key in buffer.keys():
#         batch[key] = np.array(buffer[key])[batch_indices]
#     mini_batch.append(batch)
#
# for idx, _batch in enumerate(mini_batch):
#     if idx == 0 or idx == len(mini_batch) - 2:
#         print(len(_batch["trajectory_ids"]))
#         print(len(np.unique(_batch["trajectory_ids"])))
#         print(_batch["states"])


# ============= TEST CODE LẤY BATCH ==============
# import torch
#
# buffer = buffer
# batch_size = len_batch
# mini_batch_dict = []
#
# # Tạo các nhóm tương ứng cho trajectory_ids với chỉ mục
# trajectory_ids = buffer["trajectory_ids"]
#
# # Lấy các giá trị duy nhất từ trajectory_ids và random hóa
# unique_trajectories = np.unique(trajectory_ids)
# # np.random.shuffle(unique_trajectories)  # Xáo trộn các giá trị duy nhất
#
# # Tạo danh sách chứa các nhóm với mỗi nhóm có len_batch giá trị duy nhất của buffer
# grouped_trajectories = [unique_trajectories[i:i + batch_size].tolist() for i in
#                         range(0, len(unique_trajectories), batch_size)]
#
# # Lấy các chỉ mục tương ứng được chia theo grouped_trajectories
# batches_indices = []
# for group in grouped_trajectories:
#     group_indices = np.where(np.isin(trajectory_ids, group))[0]  # Chỉ mục tương ứng với group
#     batches_indices.append(group_indices)
#
# # Lấy các giá trị trong buffer theo các chỉ mục đã chọn
# for batch_indices in batches_indices:
#     batch_dict = {key: np.array([]) for key in buffer.keys()}
#     for key in buffer.keys():
#         batch_dict[key] = np.array(buffer[key])[batch_indices]
#     mini_batch_dict.append(batch_dict)
#
# # ===== TẠO ĐẦU VÀO CHO VIỆC HUẤN LUYỆN THEO CÁC BATCH ĐÃ CHIA ==============
# """
# ĐẦU RA LÀ CÁC BATCH CÓ CHỨA DỮ LIỆU CỦA CÁC
# TENSOR CÓ SHAPE LÀ : [batch_size, num_samples, feature_size]
# """
# mini_batch = []
# for _batch_dict in mini_batch_dict:
#     # Khởi tạo dictionary chứa các batch đã được chia
#     batch = {}
#     # Khởi tạo giá trị lưu độ dài thực tế của từng traj trong batch
#     lengths_traj = None
#     # Duyệt qua tất cả các keys và chuyển đổi dữ liệu thành tensor
#     trajectory_ids = _batch_dict["trajectory_ids"]  # Dữ liệu trajectory_ids
#     for key in _batch_dict.keys():
#         if key != "trajectory_ids":
#             unique_trajectories = np.unique(trajectory_ids)
#             # Lấy liệu cho key hiện tại
#             data = _batch_dict[key]
#
#             # Tạo một mảng chứa mask cho mỗi trajectory_id
#             mask_list = [trajectory_ids == trajectory_id for trajectory_id in unique_trajectories]
#
#             # Dùng broadcasting và slicing để nhóm dữ liệu theo trajectory_id
#             grouped_data = [data[mask] for mask in mask_list]
#
#             # Tìm số mẫu lớn nhất trong mỗi nhóm để padding
#             max_samples = max(len(group) for group in grouped_data)
#
#             # lengths_traj: Danh sách độ dài thực của từng trajectory trong batch
#             lengths_traj = torch.tensor([len(group) for group in grouped_data])
#
#             # Padding dữ liệu sao cho tất cả các nhóm đều có số lượng mẫu như nhau
#             padded_data = np.array(
#                 [np.pad(group, ((0, max_samples - len(group)), (0, 0)), mode='constant') for group in grouped_data])
#
#             # Chuyển thành tensor
#             batch[key] = torch.tensor(padded_data)
#             batch["lengths_traj"] = lengths_traj
#             mini_batch.append(batch)
#
# for idx, _batch in enumerate(mini_batch):
#     if idx == 0 or idx == len(mini_batch) - 2:
#         print(len(_batch["states"]))
#         print(_batch["states"])
#         print(_batch["lengths_traj"])
