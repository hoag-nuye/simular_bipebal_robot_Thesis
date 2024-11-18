import numpy as np

def validate_add_state(array: np.ndarray, expected_shape: tuple, name: str):
    """
    Kiểm tra xem mảng đầu vào có đúng kích thước không.

    Args:
        array (np.ndarray): Mảng cần kiểm tra.
        expected_shape (tuple): Kích thước mong đợi (vd: (3,), (10,)).
        name (str): Tên của tham số để in thông báo lỗi nếu sai.

    Raises:
        ValueError: Nếu kích thước mảng không đúng.
    """
    if array.shape != expected_shape:
        raise ValueError(
            f"Tham số '{name}' có kích thước {array.shape}, nhưng kích thước mong đợi là {expected_shape}."
        )