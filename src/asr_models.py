from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


def conv_subsample_lengths(lengths: torch.Tensor, n_fft: int = 400, hop_length: int = 160) -> torch.Tensor:
    frames = torch.div(torch.clamp(lengths - n_fft, min=0), hop_length, rounding_mode="floor") + 1
    frames = torch.div(frames + 1, 2, rounding_mode="floor")
    frames = torch.div(frames + 1, 2, rounding_mode="floor")
    return torch.clamp(frames, min=1)


class CRNNCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        conv_channels: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, conv_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels // 2, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
        )
        freq_after = (n_mels + 1) // 2
        freq_after = (freq_after + 1) // 2
        rnn_input = conv_channels * freq_after
        self.encoder = nn.GRU(
            input_size=rnn_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        with torch.no_grad():
            self.classifier.bias.zero_()
            self.classifier.bias[0] = -2.0

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.size(1) < self.n_fft:
            waveforms = F.pad(waveforms, (0, self.n_fft - waveforms.size(1)))
        features = self.mel(waveforms)
        features = self.db(features)
        features = torch.clamp(features, min=-80.0, max=80.0)
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return (features - mean) / std

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(waveforms)
        x = features.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        batch, time_steps, channels, freq = x.shape
        x = x.view(batch, time_steps, channels * freq)
        encoded, _ = self.encoder(x)
        logits = self.classifier(encoded)
        output_lengths = conv_subsample_lengths(
            waveform_lengths.to(device=logits.device),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        output_lengths = torch.clamp(output_lengths, max=logits.size(1))
        return logits, output_lengths


def build_asr_model(
    model_name: str,
    vocab_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    if model_name == "crnn_ctc":
        return CRNNCTC(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unsupported ASR model: {model_name}")
