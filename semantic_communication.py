import torch
import torch.nn as nn  # 确保这行存在
from KANConv import KAN_Convolutional_Layer, KANLinear
import torch.nn.functional as F  # 通常缩写为F
import math

class semantic:
    # AWGN channel
    def AWGN_channel(x, snr, distance=1):  # used to simulate additive white gaussian noise channel
        [batch_size, len_feature] = x.shape
        x_power = torch.sum(torch.abs(x))/ (batch_size * len_feature)
        n_power = x_power / (10 ** (snr / 10.0))
        noise = torch.rand(batch_size, len_feature, device=x.device) *n_power
        H = 1/(distance*distance)
        return H*x + noise

    # Rayleigh fading channel
    def rayleigh_channel(x, snr, distance=1):
        """
        Simulate Rayleigh fading channel with additive white Gaussian noise for real-valued signals

        Parameters:
            x (torch.Tensor): Input real-valued signal tensor with shape [batch_size, length, len_feature]
            snr (float): Signal-to-Noise Ratio in dB
            distance (float): Distance between transmitter and receiver (affects path loss)

        Returns:
            torch.Tensor: Received real-valued signal after Rayleigh fading and AWGN
        """
        [batch_size, len_feature] = x.shape

        # Calculate signal power
        x_power = torch.sum(x ** 2) / (batch_size * len_feature)

        # Calculate noise power based on SNR
        n_power = x_power / (10 ** (snr / 10.0))

        # Generate Rayleigh fading coefficients (real part only)
        # Rayleigh fading is the magnitude of complex Gaussian, but for real signals we can use the real part
        H = torch.randn(batch_size, len_feature, device=x.device) * (1 / math.sqrt(2))

        # Path loss model (1/d^2 path loss)
        path_loss = 1 / (distance ** 2)

        # Apply channel effects
        faded_signal = H * x # * path_loss

        # Add real-valued AWGN
        noise = torch.randn(batch_size, len_feature, device=x.device) * math.sqrt(n_power)

        return faded_signal + noise

class SemCom_linear(nn.Module):
    def __init__(self):
        super().__init__()
        # 语义编码器
        self.se = nn.Linear(384,128)
        # 信道编码器
        self.ce = nn.Linear(128, 32)
        self.norm = nn.BatchNorm1d(32)

        # 信道解码器
        self.cd = nn.Linear(32, 128)
        # 语义解码器
        self.sd = nn.Linear(128, 384)
        # self.sd_bodies_candidate = nn.Linear(128, 36)
        # self.sd_bodies_score = nn.Linear(128, 18)
        # self.sd_hands = nn.Linear(128, 84)
        # self.sd_hands_score = nn.Linear(128, 42)
        # self.sd_faces = nn.Linear(128, 136)
        # self.sd_faces_score = nn.Linear(128, 68)


    def forward(self, x, snr):
        x = F.relu(self.se(x))
        x = self.norm(self.ce(x))
        # wireless communication
        x = semantic.AWGN_channel(x, snr)
        # x = rayleigh_channel(x,snr)
        y = F.relu(self.cd(x))
        y = F.tanh(self.sd(y))
        return y

    def SC(model, samples, device="cuda", snr=20):
        samples = torch.tensor(samples, dtype=torch.float32).to(device)  # shape: (N, D)
        model.eval()
        with torch.no_grad():
            outputs = model(samples,snr)
        outputs = outputs.cpu().numpy()
        return outputs

class SemCom_KAN(nn.Module):
    def __init__(self):
        super().__init__()
        # 语义编码器
        self.se = KANLinear(384, 128)
        # 信道编码器
        self.ce = KANLinear(128, 32)
        self.norm = nn.BatchNorm1d(32)

        # 信道解码器
        self.cd = KANLinear(32, 128)
        # 语义解码器
        self.sd = KANLinear(128, 384)



    def forward(self, x, snr):
        x = F.relu(self.se(x))
        x = self.norm(self.ce(x))
        # wireless communication
        x = semantic.AWGN_channel(x, snr)
        # encoded = rayleigh_channel(encoded,snr)
        y = F.relu(self.cd(x))
        y = F.tanh(self.sd(y))
        return y