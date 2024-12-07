# === Information ================================================================
#  @author:       Hoàng Nguyên
#  Email:         nguyen80o08nguyen@gmail.com
#  Github:        hoag.nuye
#  Created Date:  2024-12-07
# === Information ================================================================

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_param_process(path_dir, update_interval=2000):
    # Lưu trữ lịch sử các tham số
    epochs = []
    rewards_history = []
    entropy_history = []
    actor_loss_history = []
    critic_loss_history = []

    # Vẽ biểu đồ
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Khởi tạo các biểu đồ
    lines = {
        "rewards": axs[0, 0].plot([], [], label="Reward", color='blue')[0],
        "entropy": axs[0, 1].plot([], [], label="Entropy", color='orange')[0],
        "actor_loss": axs[1, 0].plot([], [], label="Actor Loss", color='green')[0],
        "critic_loss": axs[1, 1].plot([], [], label="Critic Loss", color='red')[0],
    }

    # Thiết lập tiêu đề và nhãn cho các biểu đồ
    axs[0, 0].set_title("Reward History")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Reward")

    axs[0, 1].set_title("Entropy History")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Entropy")

    axs[1, 0].set_title("Actor Loss History")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Loss")

    axs[1, 1].set_title("Critic Loss History")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss")

    plt.tight_layout()

    # Hàm khởi tạo để reset dữ liệu trước khi animation bắt đầu
    def init():
        for line in lines.values():
            line.set_data([], [])
        return lines.values()

    # Hàm cập nhật dữ liệu
    def update(frame):
        try:
            # Đọc dữ liệu từ file (dữ liệu mới nhất)
            metrics_all = torch.load(f"{path_dir}training_metrics_all.pth", weights_only=True)

            # Trích xuất dữ liệu
            new_epochs = [metrics["epochs_history"] for metrics in metrics_all]
            new_rewards = [metrics["rewards_history"] for metrics in metrics_all]
            new_entropy = [metrics["entropy_history"] for metrics in metrics_all]
            new_actor_loss = [metrics["actor_loss_history"] for metrics in metrics_all]
            new_critic_loss = [metrics["critic_loss_history"] for metrics in metrics_all]

            # Cập nhật dữ liệu cho các đường đồ thị
            lines["rewards"].set_data(new_epochs, new_rewards)
            lines["entropy"].set_data(new_epochs, new_entropy)
            lines["actor_loss"].set_data(new_epochs, new_actor_loss)
            lines["critic_loss"].set_data(new_epochs, new_critic_loss)

            # Điều chỉnh giới hạn trục x và y dựa trên dữ liệu
            for ax in axs.flatten():
                ax.relim()
                ax.autoscale_view()

        except Exception as e:
            print(f"Lỗi khi đọc dữ liệu hoặc cập nhật đồ thị: {e}")

        return lines.values()

    # Tạo animation
    ani = animation.FuncAnimation(
        fig, update,
        init_func=init,
        blit=False, interval=update_interval,
        cache_frame_data=False  # Tắt bộ nhớ đệm
    )

    plt.show()

# Ví dụ sử dụng
# plot_param_process('path/to/your/directory/', update_interval=1000)
