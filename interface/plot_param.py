import torch
import matplotlib.pyplot as plt


def plot_param_process(path_dir):
    epochs = []
    rewards_history = []
    entropy_history = []
    actor_loss_history = []
    critic_loss_history = []
    # Load dữ liệu lịch sử
    metrics_all = torch.load(f"{path_dir}training_metrics_all.pth",  weights_only=True)
    for metrics in metrics_all:
        epochs.append(metrics["epochs_history"])
        rewards_history.append(metrics["rewards_history"])
        entropy_history.append(metrics["entropy_history"])
        actor_loss_history.append(metrics["actor_loss_history"])
        critic_loss_history.append(metrics["critic_loss_history"])

    print(epochs)
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, rewards_history, label="Reward")
    plt.title("Reward History")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, entropy_history, label="Entropy", color='orange')
    plt.title("Entropy History")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")

    plt.subplot(2, 2, 3)
    plt.plot(epochs, actor_loss_history, label="Actor Loss", color='green')
    plt.title("Actor Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(epochs, critic_loss_history, label="Critic Loss", color='red')
    plt.title("Critic Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()
