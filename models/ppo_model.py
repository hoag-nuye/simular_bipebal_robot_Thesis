import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist


# Custom PPO Clipping Loss
# Hàm tính toán tổn thất PPO với clipping
def ppo_clipping_loss(y_true, y_pred, advantages, old_predictions, clip_ratio=0.2):
    ratio = torch.exp(torch.log(y_pred + 1e-10) - torch.log(old_predictions + 1e-10))  # Tính tỷ lệ cập nhật
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)  # Áp dụng clipping để tránh cập nhật vượt mức
    # Tổn thất: Giá trị nhỏ nhất giữa (tỷ lệ không clipping) và (tỷ lệ đã clipping)
    loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
    return loss


# Custom Critic Loss (Mean Squared Error)
# Hàm tính toán tổn thất cho Critic (hàm giá trị)
def critic_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


# Mô hình Actor sử dụng LSTM
class Actor(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.2):
        self.input_size = input_size
        self.output_size = output_size
        super(Actor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)  # Lớp LSTM đầu tiên
        self.norm1 = nn.LayerNorm(128)  # Chuẩn hóa layer để tăng ổn định
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout để tránh overfitting
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)  # Lớp LSTM thứ hai
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(128, 128)  # Giảm chiều đầu ra nhưng vì đầu ra là 80 nên không cần giảm
        self.output_layer = nn.Linear(128, output_size)  # Dự đoán giá trị đầu ra cuối cùng

    def forward(self, x):
        x, _ = self.lstm1(x)  # Đầu vào qua LSTM1
        x = self.norm1(x)  # Chuẩn hóa
        x = self.dropout1(x)  # Dropout
        x, _ = self.lstm2(x)  # Đầu vào qua LSTM2
        x = self.norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.projection(x))  # Áp dụng hàm kích hoạt ReLU
        return torch.softmax(self.output_layer(x), dim=-1)  # Chuẩn hóa xác suất đầu ra

    # Hàm huấn luyện n bactch size
    def train(self):
        pass


# Mô hình Critic sử dụng LSTM
class Critic(nn.Module):
    def __init__(self, input_size, dropout_rate=0.2):
        super(Critic, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.norm1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)  # Critic chỉ có 1 giá trị đầu ra (hàm giá trị)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = torch.relu(self.projection(x))
        return self.output_layer(x)  # Giá trị dự đoán cuối cùng

    # Hàm huấn luyện luyện n bactch size
    def train(self):
        pass
